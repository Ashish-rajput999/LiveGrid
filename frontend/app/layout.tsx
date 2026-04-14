import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LiveGrid ⚡ — Real-Time Power Grid Dashboard",
  description:
    "Live monitoring dashboard for power grid failure prediction using LSTM neural networks. Track node health, cascading failures, and risk scores in real-time.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="antialiased min-h-screen" style={{ fontFamily: "'Inter', sans-serif" }}>
        {children}
      </body>
    </html>
  );
}
