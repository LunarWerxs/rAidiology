var CARD = "https://github.com/LunarWerxs/rAidiology/blob/main/";
function cell(tr, text) { var td = document.createElement("td"); td.textContent = text; tr.appendChild(td); return td; }
fetch("api/status").then(function (r) { return r.json(); }).then(function (s) {
  var rows = document.getElementById("rows");
  rows.textContent = "";
  s.models.forEach(function (m) {
    var tr = document.createElement("tr");
    cell(tr, m.name);
    cell(tr, m.reads);
    cell(tr, m.installed ? "yes" : "no");
    var a = document.createElement("a");
    a.href = CARD + m.card;
    a.textContent = "Model card";
    cell(tr, "").appendChild(a);
    rows.appendChild(tr);
  });
  var gb = s.disk_free_bytes ? (s.disk_free_bytes / 1e9).toFixed(1) + " GB free" : "";
  document.getElementById("info").textContent =
    [s.version && "Version " + s.version, "Python " + s.python, s.os, gb].filter(Boolean).join(" | ");
}).catch(function () {
  document.getElementById("rows").innerHTML = "<tr><td colspan='4'>Status is not available.</td></tr>";
});
document.getElementById("stop").addEventListener("click", function () {
  fetch("api/stop", { method: "POST" }).then(function () {
    document.getElementById("stopped").textContent = "Stopped. You can close this tab.";
    document.getElementById("stop").disabled = true;
  });
});
