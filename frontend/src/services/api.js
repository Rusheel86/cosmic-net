import axios from "axios";

const BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api/v1";

const api = axios.create({ baseURL: BASE });

export const predictMass = (file) => {
  const form = new FormData();
  form.append("file", file);
  return api.post("/predict", form);
};

export const explainPrediction = (file) => {
  const form = new FormData();
  form.append("file", file);
  return api.post("/explain", form);
};

export const virialCheck = (file) => {
  const form = new FormData();
  form.append("file", file);
  return api.post("/virial", form);
};

export const dataOverview = (file) => {
  const form = new FormData();
  form.append("file", file);
  return api.post("/overview", form);
};
