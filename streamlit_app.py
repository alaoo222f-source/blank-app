
""" Streamlit AI IDE using OpenRouter API

Features

Upload a project (any mix of .py/.js/.ts/.json/.md/.txt/.html/.css, images, .pdf, .docx, etc.).

Extracts text context (directly from text files; from PDFs via PyPDF2; from DOCX via python-docx; optional OCR for images/PDF scans if pytesseract is installed).

Chat with an "ideal programmer" assistant that sees condensed context from your files.

The assistant can propose file changes using fenced blocks like:

...full file contents...

Click a button to apply those changes to a working directory, effectively acting like a lightweight IDE.

Browse/view/edit files inside the app, and download the whole project as a ZIP.


Setup

Python 3.10+

pip install streamlit requests PyPDF2 python-docx pillow pytesseract (the last two are optional; OCR requires a system tesseract binary)

Set your OpenRouter API key in one of these ways:

1. Streamlit secrets: create .streamlit/secrets.toml with OPENROUTER_API_KEY = "..."


2. Environment variable: export OPENROUTER_API_KEY=...




Run

streamlit run app.py


Notes

Models: you can try openrouter/auto, anthropic/claude-3.5-sonnet, openai/gpt-4o-mini, google/gemini-1.5-pro, etc. Availability depends on your OpenRouter account.

The app keeps a compact summary of uploaded files to stay within context limits. You can also paste specific file contents into chat using the built-in viewer. """

from __future__ import annotations
import os
import io
import re
import json
import time
import zipfile
import base64
from pathlib import Path
from typing import List, Dict, Tuple

SUPPORTED_TEXT_EXTS = { ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".kt", ".go", ".rs", ".cpp", ".c", ".cs", ".html", ".css", ".json", ".yaml", ".yml", ".toml", ".md", ".txt", ".csv", ".env", } SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

---------- Utilities ----------

def get_api_key() -> str | None: # secrets take priority, then env var, then sidebar input key = st.secrets.get("OPENROUTER_API_KEY") if hasattr(st, "secrets") else None if not key: key = os.getenv("OPENROUTER_API_KEY") return key

def truncate_tokens(text: str, max_chars: int = 12000) -> str: # crude but robust truncation by char length if len(text) <= max_chars: return text head = text[: max_chars // 2] tail = text[-max_chars // 2 :] return head + "\n\n... [TRUNCATED] ...\n\n" + tail

def extract_text_from_pdf(fbytes: bytes) -> str: if not PdfReader: return "[PDF parsing unavailable: install PyPDF2]" try: reader = PdfReader(io.BytesIO(fbytes)) parts = [] for i, page in enumerate(reader.pages): try: parts.append(page.extract_text() or "") except Exception: parts.append("") text = "\n\n".join(parts) if not text.strip() and pytesseract and Image: # fallback OCR per page image parts = [] for i, page in enumerate(reader.pages): try: # Render page to image isn't trivial without pdf2image; we skip full OCR for simplicity parts.append("[OCR skipped: install pdf2image + poppler for page rendering]") except Exception: parts.append("") text = "\n".join(parts) return text.strip() except Exception as e: return f"[PDF read error: {e}]"

def extract_text_from_docx(fbytes: bytes) -> str: if not docx: return "[DOCX parsing unavailable: install python-docx]" try: file_like = io.BytesIO(fbytes) document = docx.Document(file_like) return "\n".join(p.text for p in document.paragraphs) except Exception as e: return f"[DOCX read error: {e}]"

def ocr_image(fbytes: bytes) -> str: if not (pytesseract and Image): return "[OCR unavailable: install pillow + pytesseract + tesseract binary]" try: img = Image.open(io.BytesIO(fbytes)) return pytesseract.image_to_string(img) except Exception as e: return f"[OCR error: {e}]"

def extract_text_from_upload(name: str, fbytes: bytes) -> Tuple[str, str]: """Returns (kind, text) where kind gives how we extracted.""" suffix = Path(name).suffix.lower() if suffix in SUPPORTED_TEXT_EXTS: try: text = fbytes.decode("utf-8", errors="replace") except Exception: text = fbytes.decode("latin-1", errors="replace") return ("text", text) if suffix == ".pdf": return ("pdf", extract_text_from_pdf(fbytes)) if suffix == ".docx": return ("docx", extract_text_from_docx(fbytes)) if suffix in SUPPORTED_IMAGE_EXTS: return ("image", ocr_image(fbytes)) # Unknown binary types -> describe only return ("binary", f"[Unsupported file type {suffix}; using filename only as context]")

def summarize_file(name: str, text: str, max_chars: int = 4000) -> str: short = truncate_tokens(text, max_chars=max_chars) return f"# File: {name}\n\n{short}\n"

def ensure_workspace(root: Path) -> None: root.mkdir(parents=True, exist_ok=True)

def write_file(root: Path, rel_path: str, content: str) -> None: dest = root / rel_path dest.parent.mkdir(parents=True, exist_ok=True) dest.write_text(content, encoding="utf-8")

def parse_file_blocks(answer: str) -> List[Tuple[str, str]]: """Parse file:path ...content...  blocks and return list of (path, content). """ pattern = re.compile(r"file:([^\n]+)\n(.*?)", re.DOTALL) blocks = [] for m in pattern.finditer(answer): path = m.group(1).strip() content = m.group(2) blocks.append((path, content)) return blocks

def zip_workspace(root: Path) -> bytes: mem = io.BytesIO() with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf: for path in root.rglob("*"): if path.is_file(): zf.write(path, arcname=path.relative_to(root).as_posix()) mem.seek(0) return mem.read()

---------- OpenRouter Chat ----------

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_SYSTEM_PROMPT = ( """ أنت مساعد برمجي مثالي يعمل كبيئة تطوير مصغّرة (IDE) داخل Streamlit. المهام:

1. فهم مشروع المستخدم من الملفات المرفوعة (تجاهل الأسرار مثل مفاتيح API إذا ظهرت).


2. اقترح، أنشئ، وعدّل ملفات المشروع بإخراجها داخل كتل على الشكل:



(ضع هنا كامل محتوى الملف)

اكتب المحتوى الكامل للملف كما تريد أن يصبح بعد التعديل.

أنشئ هيكل المجلدات عند الحاجة (مثال: src/, app/, components/).

لا تختصر بأجزاء؛ اكتب الملف كاملاً ليتمكن التطبيق من حفظه كما هو.

عند إنشاء مشروع جديد، ابدأ بملف README.md يشرح التشغيل.


3. عندما تطلب ملفات إضافية موجودة لدى المستخدم، اذكر المسار المطلوب بوضوح.


4. حافظ على ردود مختصرة مفيدة، ثم أضف الكتل الخاصة بالملفات في النهاية.


5. إن احتجت تهيئة بيئة Python/JS فاكتب أيضًا requirements.txt أو package.json ضمن كتل الملفات. """ ).strip()



def call_openrouter(api_key: str, model: str, messages: List[Dict], temperature: float = 0.2, max_tokens: int = 2000) -> str: headers = { "Authorization": f"Bearer {api_key}", "HTTP-Referer": "https://streamlit-ai-ide.local/",  # put your deployed URL if you have one "X-Title": "Streamlit AI IDE", "Content-Type": "application/json", } payload = { "model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, } resp = requests.post(OPENROUTER_ENDPOINT, headers=headers, json=payload, timeout=120) if resp.status_code != 200: raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text[:300]}") data = resp.json() try: return data["choices"][0]["message"]["content"] except Exception: return json.dumps(data, ensure_ascii=False, indent=2)

---------- Streamlit UI ----------

st.set_page_config(page_title="AI IDE (OpenRouter)", layout="wide")

if "messages" not in st.session_state: st.session_state.messages = []  # chat history if "file_summaries" not in st.session_state: st.session_state.file_summaries = []  # list of strings if "uploads" not in st.session_state: st.session_state.uploads = {}  # name -> bytes if "workspace" not in st.session_state: st.session_state.workspace = str(Path.cwd() / "workspace" / "project")

st.title("🧠 AI IDE via OpenRouter")

with st.sidebar: st.header("⚙️ الإعدادات") api_key = st.text_input("OpenRouter API Key", value=get_api_key() or "", type="password") model = st.selectbox( "اختر النموذج", [ "openrouter/auto", "anthropic/claude-3.5-sonnet", "openai/gpt-4o-mini", "google/gemini-1.5-pro", "meta-llama/llama-3.1-70b-instruct", ], index=0, ) temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05) project_name = st.text_input("اسم المشروع", value="my-ai-project")

# Workspace root
workspace_root = Path("workspace") / project_name
st.session_state.workspace = str(workspace_root)
ensure_workspace(workspace_root)

st.markdown("---")
st.caption("تحميل ملفات المشروع")
up_files = st.file_uploader(
    "ارفع ملفات المشروع (متعددة)", accept_multiple_files=True,
    type=None,  # allow all; we'll filter
)
if up_files:
    for uf in up_files:
        content = uf.read()
        st.session_state.uploads[uf.name] = content
    st.success(f"تم تحميل {len(up_files)} ملف/ملفات.")

if st.button("🧮 تحليل الملفات وبناء السياق"):
    summaries = []
    for name, data in st.session_state.uploads.items():
        kind, text = extract_text_from_upload(name, data)
        if kind in ("text", "pdf", "docx", "image"):
            summaries.append(summarize_file(name, text))
        else:
            summaries.append(f"# File: {name}\n\n[Binary or unsupported file; لا يوجد نص مستخلص]\n")
    st.session_state.file_summaries = summaries
    st.success("تم إنشاء ملخصات السياق.")

if st.button("🗂️ عرض شجرة المشروع"):
    tree = []
    for p in workspace_root.rglob("*"):
        tree.append(p.relative_to(workspace_root).as_posix())
    if not tree:
        st.info("المجلد فارغ حتى الآن.")
    else:
        st.text("\n".join(sorted(tree)))

if st.button("📦 تنزيل المشروع كـ ZIP"):
    data = zip_workspace(workspace_root)
    b64 = base64.b64encode(data).decode("utf-8")
    href = f'<a download="{project_name}.zip" href="data:application/zip;base64,{b64}">تحميل الملف المضغوط</a>'
    st.markdown(href, unsafe_allow_html=True)

st.markdown("---")
st.caption("📝 عرض/تحرير ملف")
rel_path = st.text_input("مسار الملف داخل المشروع", value="README.md")
full_path = workspace_root / rel_path
edited = st.text_area("محتوى الملف", value=(full_path.read_text(encoding="utf-8") if full_path.exists() else ""), height=240)
colA, colB = st.columns(2)
with colA:
    if st.button("💾 حفظ الملف"):
        write_file(workspace_root, rel_path, edited)
        st.success("تم الحفظ.")
with colB:
    if st.button("🗑️ حذف الملف"):
        try:
            (workspace_root / rel_path).unlink(missing_ok=True)
            st.success("تم الحذف.")
        except Exception as e:
            st.error(str(e))

st.markdown("---")

---------- Chat Area ----------

st.subheader("💬 الدردشة مع المبرمج المثالي")

Show uploaded file names & context toggle

with st.expander("عرض ملخصات الملفات (السياق)"): if st.session_state.file_summaries: for s in st.session_state.file_summaries: st.markdown(f"") st.text(s) st.markdown(f"") else: st.info("لم يتم إنشاء ملخصات بعد. استخدم زر "تحليل الملفات" في الشريط الجانبي.")

for m in st.session_state.messages: with st.chat_message(m["role"]): st.markdown(m["content"])

prompt = st.chat_input("اكتب رسالتك للمساعد...")

if prompt: st.session_state.messages.append({"role": "user", "content": prompt})

# Build messages for API
if not api_key:
    st.error("يرجى إدخال OpenRouter API Key أولاً من الشريط الجانبي.")
else:
    context_blob = "\n\n".join(st.session_state.file_summaries)
    system_prompt = DEFAULT_SYSTEM_PROMPT + "\n\n" + (
        "ملخص الملفات المرفوعة:\n" + truncate_tokens(context_blob, 24000)
        if context_blob else ""
    )
    chat_messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages

    with st.spinner("يجري توليد الرد..."):
        try:
            answer = call_openrouter(api_key, model, chat_messages, temperature=temperature)
        except Exception as e:
            answer = f"حدث خطأ أثناء الاتصال بـ OpenRouter: {e}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Parse potential file blocks and offer to apply
    blocks = parse_file_blocks(answer)
    if blocks:
        st.success(f"تم الكشف عن {len(blocks)} ملف/ملفات مقترَحة في ردّ المساعد.")
        if st.button("✍️ تطبيق تعديلات الملفات المقترَحة"):
            applied = []
            for path, content in blocks:
                write_file(Path(st.session_state.workspace), path, content)
                applied.append(path)
            st.info("تم حفظ الملفات التالية:\n" + "\n".join(applied))
    else:
        st.caption("لا توجد كتل ملفات في هذا الرد. اطلب من المساعد أن يكتب الملفات داخل كتل `file:`.")

Footer hint

st.markdown( """ --- نصيحة: اطلب من المساعد مثلاً: "أنشئ مشروع FastAPI بسيط مع واجهة React، واكتب التعليمات في README، وضع كل ملف ضمن كتل file:". """ )

