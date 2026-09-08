import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Set Your Own Climb",
  description:
    "Build a custom Kilter Board climb and get a real V-grade prediction from the actual trained model, running live in your browser.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
