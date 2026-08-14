export function MetricCard({ label, value, detail }: { label: string; value: string; detail?: string }) {
  return <div className="glass rounded-2xl p-4"><p className="text-xs font-medium uppercase tracking-wider text-zinc-500">{label}</p><p className="mt-2 text-xl font-semibold tracking-tight">{value}</p>{detail && <p className="mt-1 text-xs text-zinc-500">{detail}</p>}</div>;
}
