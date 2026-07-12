import type { HealthResponse } from "./types";

export interface HealthQueryState {
  isError: boolean;
  isPending: boolean;
  data?: Pick<HealthResponse, "status" | "providersConfigured">;
}

export function getHealthActionState(health: HealthQueryState) {
  const serviceUnavailable = health.isError || health.data?.status === "unavailable";
  const generationUnavailable = serviceUnavailable || health.isPending || health.data?.providersConfigured === false;
  return { serviceUnavailable, generationUnavailable };
}
