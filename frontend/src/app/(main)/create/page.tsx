import { headers } from "next/headers";
import Link from "next/link";
import { redirect } from "next/navigation";
import CreateSong from "~/components/create";
import { SongPanel } from "~/components/create/song-panel";
import { auth } from "~/lib/auth";

export default async function HomePage() {
  const session = await auth.api.getSession({
    headers: await headers(),
  });

  if (!session) {
    redirect("/auth/sign-in");
  }

  return (
    <div className="flex h-full flex-col lg:flex-row">
      <SongPanel />
    </div>
  );
}
