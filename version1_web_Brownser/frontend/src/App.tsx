import { ChangeEvent, FormEvent, useEffect, useRef, useState } from "react";
import {
  Bot, MessageSquare, FileText, Settings, Info, Paperclip, Send,
  Plus, LoaderCircle, Sparkles,
  AlertCircle, RefreshCw
} from "lucide-react";

const API = "http://127.0.0.1:8000";

type Source = { name: string; page: number | null; distance: number; excerpt: string };
type Message = { role: "user" | "assistant"; content: string; sources?: Source[] };
type DocumentRecord = { id: string; name: string; chunks: number; size: number; status: string };

export default function App() {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState("");
  const [apiReady, setApiReady] = useState(false);
  const [activeNav, setActiveNav] = useState("Chat");
  const [showSources, setShowSources] = useState<number | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  async function refreshDocuments() {
    try {
      const res = await fetch(`${API}/api/documents`);
      if (!res.ok) throw new Error("Backend unavailable");
      const data = await res.json();
      setDocuments(data.documents ?? []);
      setApiReady(true);
    } catch {
      setApiReady(false);
    }
  }

  useEffect(() => { void refreshDocuments(); }, []);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, asking]);

  async function handleFiles(event: ChangeEvent<HTMLInputElement>) {
    const files = event.target.files;
    if (!files?.length) return;
    setUploading(true);
    setError("");
    const form = new FormData();
    Array.from(files).forEach(file => form.append("files", file));
    try {
      const res = await fetch(`${API}/api/upload`, { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed");
      setDocuments((current) => [...current, ...(data.uploaded ?? [])]);
      setMessages((current) => [...current, {
        role: "assistant",
        content: `Successfully indexed ${data.uploaded.length} document(s). You can now ask questions about their contents.`
      }]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
      await refreshDocuments();
    }
  }

  async function sendQuestion(event?: FormEvent) {
    event?.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || asking) return;
    setError("");
    setQuestion("");
    setMessages(current => [...current, { role: "user", content: trimmed }]);
    setAsking(true);
    try {
      const res = await fetch(`${API}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed, history: [] }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Could not get an answer");
      setMessages(current => [...current, {
        role: "assistant",
        content: data.answer,
        sources: data.sources ?? [],
      }]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Chat request failed");
    } finally {
      setAsking(false);
    }
  }

  function startNewChat() {
    setMessages([]);
    setError("");
    setActiveNav("Chat");
  }

  const indexedChunks = documents.reduce((sum, doc) => sum + doc.chunks, 0);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Bot size={25} strokeWidth={2.4} /></div>
          <div>
            <div className="brand-title">Local RAG Assistant</div>
            <div className="brand-subtitle">Your private document AI</div>
          </div>
        </div>

        <button className="new-chat" onClick={startNewChat}><Plus size={17} /> New chat</button>

        <nav className="navigation">
          {[
            { label: "Chat", icon: MessageSquare },
            { label: "Documents", icon: FileText },
            { label: "Settings", icon: Settings },
            { label: "About", icon: Info },
          ].map(({ label, icon: Icon }) => (
            <button key={label} className={`nav-item ${activeNav === label ? "active" : ""}`}
              onClick={() => setActiveNav(label)}>
              <Icon size={19} /> <span>{label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-divider" />
        <div className="section-label">YOUR KNOWLEDGE BASE <span>{documents.length}</span></div>
        <div className="document-list">
          {documents.map(doc => (
            <div className="document-item" key={doc.id}>
              <div className="pdf-icon"><FileText size={20} /></div>
              <div className="document-info">
                <div className="document-name" title={doc.name}>{doc.name}</div>
                <div className="document-meta"><span className="status-dot" /> {doc.status} · {doc.chunks} chunks</div>
              </div>
            </div>
          ))}
          {!documents.length && <div className="empty-docs">Your uploaded PDFs will appear here.</div>}
        </div>

        <button className="upload-sidebar" onClick={() => fileRef.current?.click()} disabled={uploading}>
          {uploading ? <LoaderCircle className="spin" size={17} /> : <Plus size={17} />}
          {uploading ? "Indexing documents…" : "Upload documents"}
        </button>
        <input ref={fileRef} type="file" accept=".pdf,application/pdf" multiple hidden onChange={handleFiles} />

        <div className="sidebar-bottom">
          <div className="ready-row">
            <span className={`status-dot ${apiReady ? "" : "offline"}`} />
            <span>{apiReady ? "Local backend connected" : "Backend disconnected"}</span>
          </div>
          <div className="system-meta">ChromaDB · Ollama · Local embeddings</div>
          <div className="system-meta">{indexedChunks} indexed chunks</div>
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div className="topbar-icon"><MessageSquare size={24} /></div>
          <div className="topbar-copy">
            <h1>{activeNav === "Chat" ? "Chat with your documents" : activeNav}</h1>
            <p>
              {activeNav === "Chat"
                ? "Ask questions and get answers grounded in your documents, with sources."
                : activeNav === "Documents"
                ? "Upload PDFs and build your searchable local knowledge base."
                : activeNav === "Settings"
                ? "This local starter uses the configured Ollama model and embedding model."
                : "A local-first Retrieval-Augmented Generation (RAG) assistant."}
            </p>
          </div>
          <button className="icon-button" title="Refresh document list" onClick={() => void refreshDocuments()}><RefreshCw size={17} /></button>
        </header>

        {activeNav === "Documents" ? (
          <section className="document-page">
            <div className="page-heading">
              <div><h2>Your documents</h2><p>PDFs are processed and indexed on this computer.</p></div>
              <button className="primary-button" onClick={() => fileRef.current?.click()}><Plus size={17} /> Upload PDF</button>
            </div>
            <div className="document-grid">
              {documents.map(doc => (
                <article className="document-card" key={doc.id}>
                  <div className="pdf-large"><FileText size={25} /></div>
                  <div className="document-name">{doc.name}</div>
                  <div className="document-meta"><span className="status-dot" /> Indexed · {doc.chunks} chunks</div>
                  <div className="document-size">{(doc.size / 1024).toFixed(0)} KB</div>
                </article>
              ))}
              {!documents.length && <div className="empty-state"><FileText size={30} /><h3>No documents yet</h3><p>Upload a PDF to start building your knowledge base.</p><button className="primary-button" onClick={() => fileRef.current?.click()}>Choose PDF</button></div>}
            </div>
          </section>
        ) : activeNav === "Settings" || activeNav === "About" ? (
          <section className="info-page">
            <div className="info-card">
              <Sparkles size={25} />
              <h2>{activeNav === "Settings" ? "Local model settings" : "About this project"}</h2>
              <p>Frontend: React + TypeScript</p>
              <p>Backend: FastAPI · PDF extraction: PyMuPDF</p>
              <p>Vector database: ChromaDB · Embeddings: Sentence Transformers</p>
              <p>Local language model: Ollama ({ "llama3.2:3b" })</p>
              <p className="muted">Change the OLLAMA_MODEL environment variable in the backend to use a different model.</p>
            </div>
          </section>
        ) : (
          <>
            <section className={`chat-area ${messages.length === 0 ? "is-empty" : ""}`}>
              {messages.length === 0 ? (
                <div className="welcome">
                  <div className="welcome-icon"><Sparkles size={27} /></div>
                  <h2>What would you like to know?</h2>
                  <p>Your documents are your knowledge base. Ask a question to find answers in them.</p>
                  <div className="suggestions">
                    {[
                      ["Summarize my documents", "Give me a concise summary of the uploaded documents."],
                      ["Find a specific detail", "What are the most important dates mentioned in my documents?"],
                      ["Ask about my resume", "What is my current job title and when did I start?"],
                    ].map(([title, prompt]) => (
                      <button className="suggestion" key={title} onClick={() => setQuestion(prompt)}>
                        <span>{title}</span><span className="suggestion-arrow">↗</span>
                      </button>
                    ))}
                  </div>
                  {!documents.length && <div className="upload-hint"><FileText size={17} /> Start by uploading a PDF from the sidebar.</div>}
                </div>
              ) : (
                <div className="messages">
                  {messages.map((message, index) => (
                    <div className={`message-row ${message.role}`} key={`${index}-${message.role}`}>
                      <div className={`avatar ${message.role}`}>
                        {message.role === "assistant" ? <Bot size={19} /> : <span>Y</span>}
                      </div>
                      <div className={`message-card ${message.role}`}>
                        <div className="message-author">{message.role === "assistant" ? "Assistant" : "You"}</div>
                        <div className="message-content">{message.content}</div>
                        {message.sources && message.sources.length > 0 && (
                          <div className="sources-area">
                            <button className="sources-toggle" onClick={() => setShowSources(showSources === index ? null : index)}>
                              <FileText size={16} /> Sources ({message.sources.length}) <span>{showSources === index ? "−" : "+"}</span>
                            </button>
                            {showSources === index && message.sources.map((source, sourceIndex) => (
                              <div className="source-card" key={`${source.name}-${source.page}-${sourceIndex}`}>
                                <div className="source-heading"><FileText size={16} /><strong>{source.name}</strong><span>Page {source.page ?? "?"}</span></div>
                                <p>{source.excerpt}</p>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                  {asking && <div className="message-row assistant"><div className="avatar assistant"><Bot size={19} /></div><div className="typing-card"><LoaderCircle className="spin" size={17} /> Searching documents and generating an answer…</div></div>}
                  <div ref={bottomRef} />
                </div>
              )}
            </section>

            {error && <div className="error-banner"><AlertCircle size={17} /><span>{error}</span><button onClick={() => setError("")}>Dismiss</button></div>}

            <div className="composer-wrap">
              <form className="composer" onSubmit={sendQuestion}>
                <button type="button" className="attach-button" title="Upload PDFs" onClick={() => fileRef.current?.click()} disabled={uploading}><Paperclip size={20} /></button>
                <textarea
                  value={question}
                  onChange={event => setQuestion(event.target.value)}
                  onKeyDown={event => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void sendQuestion(); } }}
                  placeholder="Ask a question about your documents…"
                  rows={1}
                  aria-label="Ask a question about your documents"
                />
                <button type="submit" className="send-button" disabled={!question.trim() || asking}>
                  {asking ? <LoaderCircle className="spin" size={18} /> : <Send size={18} />} <span>Send</span>
                </button>
              </form>
              <div className="composer-note">Your files and AI model run locally on your computer · Enter to send, Shift+Enter for a new line</div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
