import { createFileRoute, redirect } from "@tanstack/react-router";

export const Route = createFileRoute("/lab/$runId/")({
  beforeLoad: ({ params }) => {
    throw redirect({ to: "/lab/$runId/video", params: { runId: params.runId } });
  },
});
