import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Link, Outlet, createRootRouteWithContext } from "@tanstack/react-router";

function RootComponent() {
  const { queryClient } = Route.useRouteContext();
  return <QueryClientProvider client={queryClient}><Outlet /></QueryClientProvider>;
}

export const Route = createRootRouteWithContext<{ queryClient: QueryClient }>()({
  component: RootComponent,
  notFoundComponent: () => <main className="flex min-h-screen items-center justify-center"><div className="text-center"><h1 className="font-display text-6xl">404</h1><Link to="/" className="underline">Go home</Link></div></main>,
  errorComponent: ({ error }) => <main className="flex min-h-screen items-center justify-center px-6"><div className="max-w-md text-center"><h1 className="font-display text-3xl">This page did not load</h1><p className="mt-3 text-muted-foreground">{error.message}</p><a className="mt-5 inline-block underline" href="/">Go home</a></div></main>,
});
