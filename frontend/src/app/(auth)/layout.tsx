import "~/styles/globals.css";

import { type Metadata } from "next";
import { Geist } from "next/font/google";
import { Providers } from "~/components/providers";
import { Toaster } from "~/components/ui/sonner";

export const metadata: Metadata = {
  title: "Sonus",
  description: "Sonus",
  icons: [{ rel: "icon", url: "/favicon.ico" }],
};

const geist = Geist({
  subsets: ["latin"],
  variable: "--font-geist-sans",
});

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${geist.variable}`}>
      <body className="min-h-svh flex flex-col items-center justify-center">
        <Providers>
          {children}
          <Toaster />
        </Providers>
      </body>
    </html>
  );
}
