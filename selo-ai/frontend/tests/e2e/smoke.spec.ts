// @ts-nocheck
import { test, expect } from '@playwright/test';

// Basic smoke to ensure app loads
// Requires backend running separately if the UI fetches on load; otherwise it will still render shell.

test('loads home page', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('body')).toBeVisible();
  // If the app renders a root element, verify it exists
  const root = page.locator('#root');
  await expect(root).toBeVisible();
});
