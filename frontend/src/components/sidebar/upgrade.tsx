"use client";

import { authClient } from "~/lib/auth-client";
import { Button } from "../ui/button";

export default function Upgrade() {
  const upgrade = async () => {
    await authClient.checkout({
      products: [
        "ddf8c785-137a-42da-abd0-8fdfe0e40bd2",
        "8b90d22c-0eb0-4745-b2ce-2bd9b6893ae6",
        "2aa99c89-5653-4456-b60c-9b97bd53f30b",
      ],
    });
  };
  return (
    <Button
      variant="outline"
      size="sm"
      className="ml-2 cursor-pointer text-orange-400"
      onClick={upgrade}
    >
      Upgrade
    </Button>
  );
}
