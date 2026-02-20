import { Suspense } from "react";
import DashboardAdk from "../../../components/DashboardAdk";

export default function AdvancedDashboardPage() {
  return (
    <Suspense fallback={<div className="p-6 text-sm text-white/70">Loading dashboard...</div>}>
      <DashboardAdk />
    </Suspense>
  );
}
