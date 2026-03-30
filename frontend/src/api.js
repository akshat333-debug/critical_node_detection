/**
 * API Client for Critical Node Detection Backend
 */

const BASE = '/api';

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const error = await res.text();
    throw new Error(`API ${res.status}: ${error}`);
  }
  return res.json();
}

export async function getNetworks() {
  return request('/networks');
}

export async function analyze(params) {
  return request('/analyze', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export async function getImpact(params) {
  return request('/impact', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export async function getCascade(params) {
  return request('/cascade', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export async function getRobustness(params) {
  return request('/robustness', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export async function getTheory() {
  return request('/theory');
}
