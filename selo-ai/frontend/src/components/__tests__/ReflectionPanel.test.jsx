import React from 'react';
import { render, screen, act } from '@testing-library/react';
import '@testing-library/jest-dom';

// Mock the reflectionService used by the component
jest.mock('../../services/reflectionService', () => {
  let statusCb = null;
  let reflectionCb = null;

  return {
    reflectionService: {
      getReflections: jest.fn(async (sessionId) => {
        return [
          {
            reflection_id: 'r1',
            result: 'Initial reflection',
            reflection_type: 'message',
            created_at: new Date().toISOString(),
            user_profile_id: sessionId,
          },
        ];
      }),
      onReflectionGenerated: jest.fn((cb) => {
        reflectionCb = cb;
        return () => {};
      }),
      subscribeToConnectionStatus: jest.fn((cb) => {
        statusCb = cb;
        // Immediately report connected
        cb('connected');
        return () => {};
      }),
      disconnect: jest.fn(),
      // Test helpers
      __emitReflection: (payload) => reflectionCb && reflectionCb({ data: payload }),
      __setStatus: (s) => statusCb && statusCb(s),
    },
  };
});

import ReflectionPanel from '../ReflectionPanel';
import { reflectionService } from '../../services/reflectionService';


test('renders initial reflections and updates on new reflection', async () => {
  await act(async () => {
    render(<ReflectionPanel sessionId="test-session" messages={[]} />);
  });

  // Header and connection indicator present
  expect(screen.getByText("SELO's Inner Reflections")).toBeInTheDocument();

  // Initial item
  expect(await screen.findByText('Initial reflection')).toBeInTheDocument();

  // Simulate a new reflection via the mocked service
  await act(async () => {
    reflectionService.__emitReflection({
      reflection_id: 'r2',
      result: 'New reflection arrived',
      reflection_type: 'message',
      created_at: new Date().toISOString(),
      user_profile_id: 'test-session',
    });
  });

  expect(screen.getByText('New reflection arrived')).toBeInTheDocument();
});

test('shows connection status changes (polling and offline)', async () => {
  await act(async () => {
    render(<ReflectionPanel sessionId="test-session" messages={[]} />);
  });

  // Initially connected from mock
  expect(screen.getByText('Live')).toBeInTheDocument();

  // Switch to polling and verify label updates
  await act(async () => {
    reflectionService.__setStatus('polling');
  });
  expect(screen.getByText('Polling')).toBeInTheDocument();

  // Switch to offline and verify label updates
  await act(async () => {
    reflectionService.__setStatus('offline');
  });
  expect(screen.getByText('Offline')).toBeInTheDocument();
});

test('formats timestamps robustly (invalid and missing)', async () => {
  // Next call to getReflections returns entries with invalid and missing timestamps
  reflectionService.getReflections.mockResolvedValueOnce([
    {
      reflection_id: 'r_invalid',
      result: 'Invalid time reflection',
      reflection_type: 'message',
      created_at: 'not-a-date',
      user_profile_id: 'test-session',
    },
    {
      reflection_id: 'r_missing',
      result: 'Unknown time reflection',
      reflection_type: 'message',
      created_at: null,
      user_profile_id: 'test-session',
    },
  ]);

  await act(async () => {
    render(<ReflectionPanel sessionId="test-session" messages={[]} />);
  });

  // Verify both rows present
  expect(await screen.findByText('Invalid time reflection')).toBeInTheDocument();
  expect(await screen.findByText('Unknown time reflection')).toBeInTheDocument();

  // Verify formatted timestamp footers are rendered
  expect(screen.getByText('Invalid time')).toBeInTheDocument();
  expect(screen.getByText('Unknown time')).toBeInTheDocument();
});
