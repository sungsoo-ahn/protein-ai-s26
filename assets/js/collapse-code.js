// Collapse code blocks into <details> elements (collapsed by default).
// Targets the static .highlighter-rouge wrapper that kramdown generates,
// so this script works regardless of copy_code.js execution order.
document.querySelectorAll("div.highlighter-rouge").forEach(function (block) {
  // Extract language from the wrapper class, e.g. "language-python" → "Python"
  var lang = "Code";
  var match = block.className.match(/language-(\S+)/);
  if (match && match[1] !== "plaintext") {
    lang = match[1].charAt(0).toUpperCase() + match[1].slice(1);
  }

  var details = document.createElement("details");
  details.className = "code-collapse";

  var summary = document.createElement("summary");

  // Check if the next sibling is a caption div — if so, use its text
  // as the summary label and pull the caption inside <details>.
  var next = block.nextElementSibling;
  var caption = null;
  if (next && next.classList.contains("caption")) {
    caption = next;
  }
  summary.textContent = lang;

  // Insert <details> where the block was, then move block inside
  block.parentNode.insertBefore(details, block);
  details.appendChild(summary);
  details.appendChild(block);
  if (caption) {
    details.appendChild(caption);
  }
});
