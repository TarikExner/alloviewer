// src/App.tsx
import { Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import CDCApp from "./pages/CDCApp";
import CrossmatchApp from "./pages/CrossmatchApp";
import Contact from "./pages/Contact";
import About from "./pages/About";
import Tutorial from "./pages/Tutorial";
import FCXM from "./pages/FCXM";
import UsePolicy from "./pages/UsePolicy";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/cdc" element={<CDCApp />} />
      <Route path="/crossmatch" element={<CrossmatchApp />} />
      <Route path="/fcxm" element={<FCXM />} />
      <Route path="/contact" element={<Contact />} />
      <Route path="/about" element={<About />} />
      <Route path="/tutorial" element={<Tutorial />} />
      <Route path="/use-policy" element={<UsePolicy />} />
      <Route path="*" element={<Home />} />
    </Routes>
  );
}

export const API_BASE =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
