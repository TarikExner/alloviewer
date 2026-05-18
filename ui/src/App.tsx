// src/App.tsx
import { Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import CDCApp from "./pages/CDCApp";
import CrossmatchApp from "./pages/CrossmatchApp";
import Contact from "./pages/Contact";
import Docs from "./pages/Docs";
import About from "./pages/About";
import Tutorial from "./pages/Tutorial";
import FCXM from "./pages/FCXM";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/cdc" element={<CDCApp />} />
      <Route path="/crossmatch" element={<CrossmatchApp />} />
      <Route path="*" element={<Home />} />
      <Route path="/contact" element={<Contact />} />
      <Route path="/docs" element={<Docs />} />
      <Route path="/about" element={<About />} />
      <Route path="/tutorial" element={<Tutorial />} />
      <Route path="/fcxm" element={<FCXM />} />
    </Routes>
  );
}

export const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
