const log = document.getElementById("log"), form = document.getElementById("form");
const input = document.getElementById("input"), send = document.getElementById("send");
let history = [], system = "", busy = false;

fetch("/api/config").then(r => r.json()).then(c => {
  if (c.error) return addMsg("assistant", c.error, true);
  document.getElementById("model").textContent = c.model;
  system = c.system || "";
});

// Safe rendering: text only, with ```code``` blocks as <pre>. No innerHTML.
function render(el, text) {
  el.replaceChildren();
  text.split("```").forEach((part, i) => {
    if (i % 2) { const pre = document.createElement("pre"); pre.textContent = part.replace(/^\w*\n/, ""); el.append(pre); }
    else el.append(document.createTextNode(part));
  });
}
function addMsg(role, text, isError = false) {
  document.getElementById("empty")?.remove();
  const row = document.createElement("div"); row.className = "msg " + role;
  const b = document.createElement("div"); b.className = "bubble" + (isError ? " error" : "");
  render(b, text); row.append(b); log.append(row); log.scrollTop = log.scrollHeight;
  return b;
}

async function ask(text) {
  history.push({ role: "user", content: text });
  addMsg("user", text);
  const bubble = addMsg("assistant", ""); bubble.classList.add("typing");
  const messages = (system ? [{ role: "system", content: system }] : []).concat(history);
  let answer = "";
  try {
    const r = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ messages }) });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
    const reader = r.body.getReader(), dec = new TextDecoder(); let buf = "";
    for (;;) {
      const { done, value } = await reader.read(); if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n"); buf = lines.pop();
      for (const l of lines) {
        if (!l.startsWith("data: ") || l.includes("[DONE]")) continue;
        const delta = JSON.parse(l.slice(6)).choices?.[0]?.delta || {};
        answer += delta.content || "";
        // Gemma thinks first: show a hint until the first answer token arrives.
        bubble.classList.toggle("thinking", !answer && !!delta.reasoning);
        if (!answer && delta.reasoning) bubble.textContent = "Réflexion…"; else render(bubble, answer);
        log.scrollTop = log.scrollHeight;
      }
    }
    history.push({ role: "assistant", content: answer });
  } catch (e) {
    history.pop(); bubble.classList.add("error"); render(bubble, "Erreur : " + e.message);
  }
  bubble.classList.remove("typing");
}

form.addEventListener("submit", async e => {
  e.preventDefault();
  const text = input.value.trim(); if (!text || busy) return;
  input.value = ""; input.style.height = "auto"; busy = send.disabled = true;
  await ask(text); busy = send.disabled = false; input.focus();
});
input.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); } });
input.addEventListener("input", () => { input.style.height = "auto"; input.style.height = input.scrollHeight + "px"; });
document.querySelectorAll("#themes li").forEach(li => li.onclick = () => {
  input.value = "Que dit le groupe sur : " + li.textContent + " ?"; input.focus();
});
document.getElementById("clear").onclick = () => { history = []; location.reload(); };
