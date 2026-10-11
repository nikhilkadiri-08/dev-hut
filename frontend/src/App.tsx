import { useCallback, useEffect, useState } from "react";
import { ApiError, fetchMetadata, fetchProfile } from "./api";
import { AdminView } from "./components/AdminView";
import { AnnTrainingView } from "./components/AnnTrainingView";
import { OperatorView } from "./components/OperatorView";
import type { Metadata, ProfileResponse } from "./types";

type Role = "operator" | "admin" | "training";

const ROLE_STORAGE_KEY = "impulse-optimiser-role";

const ROLES: { id: Role; label: string }[] = [
  { id: "operator", label: "Test request" },
  { id: "admin", label: "Generator setup" },
  { id: "training", label: "Training" },
];

function storedRole(): Role {
  const saved = localStorage.getItem(ROLE_STORAGE_KEY);
  if (saved === "admin" || saved === "training") return saved;
  return "operator";
}

export function App() {
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [profile, setProfile] = useState<ProfileResponse | null>(null);
  const [role, setRole] = useState<Role>(storedRole);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([fetchMetadata(), fetchProfile()])
      .then(([loadedMetadata, loadedProfile]) => {
        setMetadata(loadedMetadata);
        setProfile(loadedProfile);
      })
      .catch((error: unknown) =>
        setLoadError(
          error instanceof ApiError ? error.message : "Could not load configuration.",
        ),
      );
  }, []);

  const chooseRole = useCallback((next: Role) => {
    setRole(next);
    localStorage.setItem(ROLE_STORAGE_KEY, next);
  }, []);

  if (loadError) {
    return (
      <main className="shell shell--centred">
        <section className="card card--error">
          <h2 className="card__title">Backend unavailable</h2>
          <p className="card__body">{loadError}</p>
        </section>
      </main>
    );
  }

  if (!metadata || !profile) {
    return (
      <main className="shell shell--centred">
        <p className="loading">Loading&hellip;</p>
      </main>
    );
  }

  return (
    <div className="shell">
      <header className="masthead">
        <div>
          <h1 className="masthead__title">Impulse generator</h1>
        </div>
        <nav className="roles" aria-label="Interface">
          {ROLES.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`role${item.id === role ? " role--active" : ""}`}
              aria-current={item.id === role}
              onClick={() => chooseRole(item.id)}
            >
              {item.label}
            </button>
          ))}
        </nav>
      </header>

      {role === "admin" ? (
        <AdminView
          metadata={metadata}
          profile={profile}
          onProfileChange={setProfile}
        />
      ) : role === "training" ? (
        <AnnTrainingView />
      ) : (
        <OperatorView
          metadata={metadata}
          profile={profile}
          onOpenAdmin={() => chooseRole("admin")}
        />
      )}
    </div>
  );
}
