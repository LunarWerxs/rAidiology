// The one read of the reduced-motion preference. Every module asks here instead of making its own
// media query, so the query and its no-window guard live in one place.
const query = typeof window !== 'undefined' ? window.matchMedia?.('(prefers-reduced-motion: reduce)') : undefined;

export function prefersReducedMotion() {
  return !!query?.matches;
}
