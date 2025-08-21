import { AccountView } from "@daveyplate/better-auth-ui";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { Button } from "~/components/ui/button";
import { accountViewPaths } from "@daveyplate/better-auth-ui/server";

export const dynamicParams = false;

export function generateStaticParams() {
  return Object.values(accountViewPaths).map((accountView) => ({
    accountView,
  }));
}

export default async function AccountPage({
  params,
}: {
  params: Promise<{ accountView: string }>;
}) {
  const { accountView } = await params;

  return (
    <div className="w-full px-4 py-4 sm:px-6">
      <div className="mb-4 pl-1">
        <Button asChild variant="outline" size="sm">
          <Link href="/" className="inline-flex items-center gap-2">
            <ArrowLeft className="h-4 w-4" />
            Back
          </Link>
        </Button>
      </div>
      <AccountView pathname={accountView} />
    </div>
  );
}
