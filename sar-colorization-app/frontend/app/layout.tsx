import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "@/components/providers";

export const metadata: Metadata = {
  title: "GeoVision | See Beyond",
  description: "See Beyond with AI-powered SAR-to-optical intelligence.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en" suppressHydrationWarning><body><Providers>{children}</Providers></body></html>;
}
