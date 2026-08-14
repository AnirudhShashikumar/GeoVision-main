"use client";

import { AnimatePresence, motion } from "framer-motion";
import { CheckCircle2, Eye, EyeOff, KeyRound, LoaderCircle, ShieldCheck, Sparkles, X } from "lucide-react";
import { useState } from "react";
import type { ProviderId } from "@/types/api";
import { saveProviderSettings } from "@/services/api";

type Props = { open: boolean; onClose: () => void; onSaved: () => void; notice?: string };

const providers: { id: ProviderId; label: string; note: string }[] = [
  { id: "openai", label: "OpenAI", note: "Available now" },
  { id: "gemini", label: "Google Gemini", note: "Future support" },
  { id: "anthropic", label: "Anthropic Claude", note: "Future support" },
];

export function AIProviderModal({ open, onClose, onSaved, notice }: Props) {
  const [provider, setProvider] = useState<ProviderId>("openai");
  const [apiKey, setApiKey] = useState("");
  const [visible, setVisible] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const valid = apiKey.trim().length > 0;
  const save = async () => {
    if (!valid) { setError("API key cannot be empty."); return; }
    setSaving(true); setError("");
    try { await saveProviderSettings(provider, apiKey); setApiKey(""); onSaved(); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Could not save configuration."); }
    finally { setSaving(false); }
  };
  return <AnimatePresence>{open && <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="fixed inset-0 z-[80] grid place-items-center overflow-y-auto bg-black/70 p-4 backdrop-blur-sm"><motion.section initial={{ opacity: 0, y: 18, scale: .98 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: 12, scale: .98 }} transition={{ duration: .2 }} role="dialog" aria-modal="true" aria-labelledby="provider-modal-title" className="w-full max-w-lg rounded-3xl border border-sky-300/20 bg-zinc-950/95 p-5 shadow-2xl shadow-sky-500/10 sm:p-7"><div className="flex items-start justify-between gap-5"><div><div className="mb-3 grid h-11 w-11 place-items-center rounded-2xl bg-sky-400/15 text-sky-200"><Sparkles size={20}/></div><h2 id="provider-modal-title" className="text-xl font-semibold">Connect AI Provider</h2><p className="mt-2 text-sm leading-6 text-zinc-400">To use AI-powered image interpretation, connect your preferred AI provider by entering your API key. Your key is stored locally and is never displayed after saving.</p></div><button onClick={onClose} className="rounded-xl p-2 text-zinc-400 hover:bg-white/10 hover:text-white" aria-label="Close configuration"><X size={18}/></button></div>{notice && <p className="mt-4 rounded-xl border border-amber-300/20 bg-amber-300/[.06] p-3 text-sm text-amber-100">{notice}</p>}<div className="mt-6"><label className="text-sm font-medium text-zinc-200">Provider</label><div className="mt-2 grid gap-2">{providers.map(item => <button key={item.id} onClick={() => setProvider(item.id)} className={`flex items-center justify-between rounded-xl border p-3 text-left transition ${provider === item.id ? "border-sky-300/50 bg-sky-400/[.10]" : "border-white/[.08] bg-white/[.025] hover:border-white/20"}`}><span className="text-sm text-zinc-100">{item.label}</span><span className={`text-xs ${item.id === "openai" ? "text-emerald-300" : "text-zinc-500"}`}>{item.note}</span></button>)}</div></div><div className="mt-5"><label htmlFor="provider-api-key" className="text-sm font-medium text-zinc-200">API Key</label><div className="mt-2 flex overflow-hidden rounded-xl border border-white/[.12] bg-white/[.03] focus-within:border-sky-300/60"><KeyRound className="m-3 shrink-0 text-zinc-500" size={18}/><input id="provider-api-key" value={apiKey} onChange={event => { setApiKey(event.target.value); setError(""); }} type={visible ? "text" : "password"} placeholder="Paste your API key..." className="min-w-0 flex-1 bg-transparent py-3 text-sm outline-none placeholder:text-zinc-600" autoComplete="off"/><button onClick={() => setVisible(value => !value)} className="px-3 text-zinc-400 hover:text-white" aria-label={visible ? "Hide API key" : "Show API key"}>{visible ? <EyeOff size={18}/> : <Eye size={18}/>}</button></div>{error && <p className="mt-2 text-sm text-rose-300">{error}</p>}</div><div className="mt-4 flex items-start gap-2 rounded-xl border border-sky-400/15 bg-sky-400/[.06] p-3 text-xs leading-5 text-sky-100/80"><ShieldCheck className="mt-0.5 shrink-0 text-sky-300" size={16}/>The browser never reads the saved key. Local macOS installs use Keychain; deployed apps use server environment variables.</div><div className="mt-5 flex flex-wrap gap-x-5 gap-y-2 text-sm"><a href="https://platform.openai.com/api-keys" target="_blank" rel="noreferrer" className="text-sky-300 hover:text-sky-200">Get OpenAI API Key</a><a href="https://aistudio.google.com/app/apikey" target="_blank" rel="noreferrer" className="text-sky-300 hover:text-sky-200">Get Gemini API Key</a></div><div className="mt-7 flex justify-end gap-3"><button onClick={onClose} disabled={saving} className="rounded-xl px-4 py-2.5 text-sm text-zinc-400 hover:bg-white/[.06] hover:text-white">Cancel</button><button onClick={save} disabled={!valid || saving} className="inline-flex items-center gap-2 rounded-xl bg-sky-300 px-4 py-2.5 text-sm font-semibold text-zinc-950 transition hover:bg-sky-200 disabled:cursor-not-allowed disabled:opacity-40">{saving ? <LoaderCircle className="animate-spin" size={16}/> : <CheckCircle2 size={16}/>}Save & Continue</button></div></motion.section></motion.div>}</AnimatePresence>;
}
