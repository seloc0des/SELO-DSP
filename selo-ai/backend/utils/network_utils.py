"""
Network utilities for SELO AI backend.
Provides dynamic network detection and configuration loading.
"""
import os
import socket
import subprocess
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger("selo.network")

def get_primary_network_ip() -> str:
    """
    Get the primary network IP address of this machine.
    Uses the same logic as the installer's deduce_host_ip() function.
    """
    try:
        # Method 1: Get IP from default route interface (most reliable)
        if _command_exists('ip'):
            # First get the default interface
            result = subprocess.run(
                ['ip', '-4', 'route', 'show', 'default'], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'default' in line:
                        parts = line.split()
                        try:
                            dev_idx = parts.index('dev')
                            if dev_idx + 1 < len(parts):
                                default_if = parts[dev_idx + 1]
                                # Now get IP of this interface
                                addr_result = subprocess.run(
                                    ['ip', '-4', 'addr', 'show', 'dev', default_if],
                                    capture_output=True,
                                    text=True,
                                    timeout=5
                                )
                                if addr_result.returncode == 0:
                                    for addr_line in addr_result.stdout.split('\n'):
                                        if 'inet ' in addr_line:
                                            addr_parts = addr_line.strip().split()
                                            for i, part in enumerate(addr_parts):
                                                if part == 'inet' and i + 1 < len(addr_parts):
                                                    ip = addr_parts[i + 1].split('/')[0]
                                                    if _is_valid_ip(ip):
                                                        logger.info(f"Detected primary IP via default interface {default_if}: {ip}")
                                                        return ip
                                                    break
                        except (ValueError, IndexError):
                            continue
    except Exception as e:
        logger.debug(f"ip route default interface method failed: {e}")

    try:
        # Method 2: Use hostname -I with preference for common network ranges
        if _command_exists('hostname'):
            result = subprocess.run(
                ['hostname', '-I'], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.returncode == 0:
                ips = result.stdout.strip().split()
                
                # First pass: prefer 192.168.* (common home/office networks)
                for ip in ips:
                    if _is_valid_ip(ip) and ip.startswith('192.168.'):
                        logger.info(f"Detected primary IP via hostname -I (192.168.*): {ip}")
                        return ip
                
                # Second pass: other private ranges
                for ip in ips:
                    if _is_valid_ip(ip) and (
                        ip.startswith('10.') or
                        any(ip.startswith(f'172.{i}.') for i in range(16, 32))
                    ):
                        logger.info(f"Detected primary IP via hostname -I (private): {ip}")
                        return ip
                
                # Third pass: any non-localhost IP
                for ip in ips:
                    if _is_valid_ip(ip) and not ip.startswith('127.'):
                        logger.info(f"Detected primary IP via hostname -I (any): {ip}")
                        return ip
    except Exception as e:
        logger.debug(f"hostname -I method failed: {e}")

    try:
        # Method 3: Connect to external address to determine local IP (fallback)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if _is_valid_ip(ip):
                logger.info(f"Detected primary IP via socket connect: {ip}")
                return ip
    except Exception as e:
        logger.debug(f"Socket connect method failed: {e}")

    # Final fallback
    logger.warning("Could not detect primary network IP, using localhost")
    return "127.0.0.1"

def load_environment_config() -> Dict[str, Any]:
    """
    Load configuration from the environment files created during installation.
    Checks both /etc/selo-ai/environment (system) and backend/.env (local).
    """
    config = {}
    
    # Load from system environment file (created by installer)
    system_env_path = "/etc/selo-ai/environment"
    if os.path.exists(system_env_path):
        try:
            with open(system_env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        config[key.strip()] = value.strip()
            logger.info(f"Loaded {len(config)} variables from {system_env_path}")
        except Exception as e:
            logger.warning(f"Could not read system environment file {system_env_path}: {e}")
    
    # Load from local backend/.env (may override system settings)
    backend_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(backend_env_path):
        try:
            local_config = {}
            with open(backend_env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        local_config[key.strip()] = value.strip()
            config.update(local_config)
            logger.info(f"Loaded {len(local_config)} variables from {backend_env_path}")
        except Exception as e:
            logger.warning(f"Could not read backend environment file {backend_env_path}: {e}")
    
    return config

def get_api_base_url(request_headers: Optional[Dict[str, str]] = None, config: Optional[Dict[str, Any]] = None) -> str:
    """
    Determine the API base URL dynamically.
    Priority:
    1. API_URL from environment files
    2. Construct from HOST_IP + SELO_AI_PORT from environment
    3. Detect from request headers (reverse proxy support)
    4. Auto-detect network IP + default port
    """
    # Load configuration from installation (use provided config to avoid duplicate loading)
    if config is None:
        config = load_environment_config()
    
    # Method 1: Direct API_URL from environment
    api_url = config.get('API_URL')
    if api_url:
        logger.info(f"Using API_URL from environment: {api_url}")
        return api_url
    
    # Method 2: Construct from HOST_IP + PORT from environment
    host_ip = config.get('HOST_IP')
    port = config.get('SELO_AI_PORT') or config.get('PORT', '8000')
    
    if host_ip and host_ip != '0.0.0.0':
        api_url = f"http://{host_ip}:{port}"
        logger.info(f"Constructed API URL from environment HOST_IP: {api_url}")
        return api_url
    
    # Method 3: Use request headers if available (reverse proxy support)
    if request_headers:
        scheme = request_headers.get("x-forwarded-proto", "http")
        forwarded_host = request_headers.get("x-forwarded-host")
        host_header = request_headers.get("host")
        
        if forwarded_host:
            api_url = f"{scheme}://{forwarded_host}"
            logger.info(f"Using forwarded host from headers: {api_url}")
            return api_url
        elif host_header:
            api_url = f"{scheme}://{host_header}"
            logger.info(f"Using host header: {api_url}")
            return api_url
    
    # Method 4: Auto-detect network IP
    detected_ip = get_primary_network_ip()
    api_url = f"http://{detected_ip}:{port}"
    logger.info(f"Auto-detected API URL: {api_url}")
    return api_url

def get_frontend_url(api_base_url: str, config: Optional[Dict[str, Any]] = None) -> str:
    """
    Determine the frontend URL based on API base URL.
    Assumes frontend runs on port 3000 on the same host as the API.
    """
    # Use provided config to avoid duplicate loading
    if config is None:
        config = load_environment_config()
    
    # Check if explicitly configured
    frontend_url = config.get('FRONTEND_URL')
    if frontend_url:
        return frontend_url
    
    # Extract host from API URL and use port 3000
    try:
        from urllib.parse import urlparse
        parsed = urlparse(api_base_url)
        frontend_url = f"{parsed.scheme}://{parsed.hostname}:3000"
        return frontend_url
    except Exception:
        # Fallback
        return api_base_url.replace(':8000', ':3000')

def _command_exists(command: str) -> bool:
    """Check if a command exists in PATH."""
    try:
        subprocess.run(['which', command], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def _is_valid_ip(ip: str) -> bool:
    """Validate if string is a valid IPv4 address."""
    try:
        socket.inet_aton(ip)
        return True
    except socket.error:
        return False
