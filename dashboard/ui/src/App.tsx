import { useEffect, useState } from "react";
import Box from "@mui/material/Box";

function App() {
  const [apiStatus, setApiStatus] = useState("checking…");

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then((d) => setApiStatus(d.status))
      .catch((e) => setApiStatus(`error: ${e.message}`));
  }, []);

  return (
    <Box component="main" sx={{ maxWidth: 720, mx: "auto", py: 6, px: 3 }}>
      <h1>olmo-eval dashboard</h1>
      <p>
        API: <code>{apiStatus}</code>
      </p>
    </Box>
  );
}

export default App;
