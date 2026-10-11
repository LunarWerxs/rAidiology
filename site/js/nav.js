// Phone menu: under 900px the main links (plus GitHub and Discord) open as a full-width panel under the header.
// Closes on a link, Escape, a click outside the header, or when the window grows past the phone layout.
const btn = document.querySelector(".menu-btn");
const nav = document.getElementById("site-nav");
if (btn && nav) {
  const set = (open) => {
    nav.classList.toggle("open", open);
    btn.setAttribute("aria-expanded", String(open));
    btn.setAttribute("aria-label", open ? "Close the menu" : "Menu");
  };
  btn.addEventListener("click", () => set(!nav.classList.contains("open")));
  nav.addEventListener("click", (e) => { if (e.target.closest("a")) set(false); });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && nav.classList.contains("open")) { set(false); btn.focus(); }
  });
  document.addEventListener("click", (e) => {
    if (nav.classList.contains("open") && !e.target.closest(".site-head")) set(false);
  });
  const wide = matchMedia("(min-width: 900px)");
  wide.addEventListener("change", () => { if (wide.matches) set(false); });
}
