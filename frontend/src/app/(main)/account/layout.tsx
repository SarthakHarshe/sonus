export default function AccountLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // Minimal pass-through layout to avoid custom styles affecting pages
  return children;
}
