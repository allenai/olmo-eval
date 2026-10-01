import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { VarnishApp } from "@allenai/varnish2/components";
import { BrowserRouter, Routes, Route } from "react-router";
import "./index.css";
import App from "./App.tsx";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <VarnishApp>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<App />} />
        </Routes>
      </BrowserRouter>
    </VarnishApp>
  </StrictMode>
);
