import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
// @ts-ignore CSS is handled by the bundler, not TypeScript.
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
