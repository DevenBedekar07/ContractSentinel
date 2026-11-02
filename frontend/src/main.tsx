import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import LandingPage from './pages/LandingPage';
import './styles.css';
import './landing-page.css';

function Root() {
  const path = window.location.pathname;
  return path === '/app' || path.startsWith('/app/') ? <App /> : <LandingPage />;
}

createRoot(document.getElementById('root')!).render(
  <React.StrictMode><Root /></React.StrictMode>,
);
