import React, { useState, useEffect } from 'react';
import {
  Layers,
  ListTodo,
  History,
  Settings,
  RefreshCw,
  Send,
  CheckCircle2,
  AlertCircle,
  Database,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  Mail,
  Lock,
  Sparkles,
  Search,
  Check,
  TrendingUp,
  X
} from 'lucide-react';

// API URL: Set VITE_API_URL in Vercel environment variables to your Railway backend URL
const API_BASE = import.meta.env.VITE_API_URL || 'https://email-ticket-classifier-production.up.railway.app';


export default function App() {
  // Navigation & Authentication
  const [activeTab, setActiveTab] = useState('test');
  const [isAuthenticated, setIsAuthenticated] = useState(true);
  const [passwordInput, setPasswordInput] = useState('');
  const [authError, setAuthError] = useState('');

  // Dashboard Stats & Model Status
  const [statusData, setStatusData] = useState({
    active_model: 'Loading...',
    stable_model: null,
    latest_run: null,
    feedback_count: 0,
    hf_repo_id: ''
  });
  const [isRefreshingStatus, setIsRefreshingStatus] = useState(false);

  // Tickets Data
  const [tickets, setTickets] = useState([]);
  const [isLoadingTickets, setIsLoadingTickets] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  // Routing Settings Data
  const [settings, setSettings] = useState({});
  const [isLoadingSettings, setIsLoadingSettings] = useState(false);
  const [isSavingSettings, setIsSavingSettings] = useState(false);

  // Available labels/categories
  const [labels, setLabels] = useState([]);

  // Test Classification State
  const [testSubject, setTestSubject] = useState('');
  const [testBody, setTestBody] = useState('');
  const [isClassifying, setIsClassifying] = useState(false);
  const [classifyResult, setClassifyResult] = useState(null);

  // Accordion UI State
  const [expandedTickets, setExpandedTickets] = useState(new Set());
  const [correctedLabels, setCorrectedLabels] = useState({});

  // Toast Notifications
  const [toasts, setToasts] = useState([]);

  // Retrieve auth password
  const getPassword = () => localStorage.getItem('dashboard_password') || '';

  // Trigger toast notifications
  const addToast = (message, type = 'info') => {
    const id = Date.now();
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4000);
  };

  // Helper fetch wrapper that adds Auth headers
  const apiFetch = async (endpoint, options = {}) => {
    const url = `${API_BASE}${endpoint}`;
    const password = getPassword();
    const headers = {
      'Content-Type': 'application/json',
      ...options.headers,
    };
    if (password) {
      headers['Authorization'] = `Bearer ${password}`;
    }

    try {
      const response = await fetch(url, { ...options, headers });
      if (response.status === 401) {
        setIsAuthenticated(false);
        localStorage.removeItem('dashboard_password');
        throw new Error('Unauthorized');
      }
      if (!response.ok) {
        const errorText = await response.text();
        let errorMsg = 'API request failed';
        try {
          const parsed = JSON.parse(errorText);
          errorMsg = parsed.detail || errorMsg;
        } catch (_) {}
        throw new Error(errorMsg);
      }
      return await response.json();
    } catch (error) {
      console.error(`API Error on ${endpoint}:`, error);
      throw error;
    }
  };

  // Validate password on submission
  const handleLogin = async (e) => {
    e.preventDefault();
    setAuthError('');
    try {
      localStorage.setItem('dashboard_password', passwordInput);
      // Try fetching status to verify password
      const data = await apiFetch('/api/status');
      setStatusData(data);
      setIsAuthenticated(true);
      addToast('Authenticated successfully', 'success');
      fetchDashboardData();
    } catch (err) {
      localStorage.removeItem('dashboard_password');
      setAuthError(err.message === 'Unauthorized' ? 'Incorrect admin password' : 'Could not connect to Railway API');
    }
  };

  // Log out
  const handleLogout = () => {
    localStorage.removeItem('dashboard_password');
    setIsAuthenticated(false);
    setPasswordInput('');
    addToast('Logged out', 'info');
  };

  // Load Status / Health
  const loadStatus = async (silent = false) => {
    if (!silent) setIsRefreshingStatus(true);
    try {
      const data = await apiFetch('/api/status');
      setStatusData(data);
    } catch (err) {
      if (err.message !== 'Unauthorized') {
        addToast('Failed to fetch CLP system status', 'error');
      }
    } finally {
      if (!silent) setIsRefreshingStatus(false);
    }
  };

  // Load tickets list
  const loadTickets = async () => {
    setIsLoadingTickets(true);
    try {
      // Load both queues depending on tab
      const statusFilter = activeTab === 'review' ? 'pending_review,auto_routed' : 'resolved';
      const data = await apiFetch(`/api/tickets?status=${statusFilter}`);
      setTickets(data);
      
      // Reset correct labels mapping
      const initialLabels = {};
      data.forEach(t => {
        initialLabels[t.id] = t.predicted_label;
      });
      setCorrectedLabels(prev => ({ ...prev, ...initialLabels }));
    } catch (err) {
      if (err.message !== 'Unauthorized') {
        addToast('Failed to load tickets', 'error');
      }
    } finally {
      setIsLoadingTickets(false);
    }
  };

  // Load Labels list
  const loadLabels = async () => {
    try {
      const data = await apiFetch('/api/labels');
      setLabels(data.labels || []);
    } catch (err) {
      console.error(err);
    }
  };

  // Load Settings
  const loadSettings = async () => {
    setIsLoadingSettings(true);
    try {
      const data = await apiFetch('/api/settings');
      setSettings(data || {});
    } catch (err) {
      if (err.message !== 'Unauthorized') {
        addToast('Failed to load routing settings', 'error');
      }
    } finally {
      setIsLoadingSettings(false);
    }
  };

  // Save Settings
  const saveSettings = async (e) => {
    e.preventDefault();
    setIsSavingSettings(true);
    try {
      await apiFetch('/api/settings', {
        method: 'POST',
        body: JSON.stringify(settings)
      });
      addToast('Routing settings saved successfully', 'success');
    } catch (err) {
      addToast('Failed to save settings', 'error');
    } finally {
      setIsSavingSettings(false);
    }
  };

  // Classify a manual test email
  const handleClassify = async (e) => {
    e.preventDefault();
    if (!testSubject && !testBody) {
      addToast('Subject or body is required', 'error');
      return;
    }
    setIsClassifying(true);
    setClassifyResult(null);
    try {
      const data = await apiFetch('/api/classify', {
        method: 'POST',
        body: JSON.stringify({ subject: testSubject, body: testBody })
      });
      setClassifyResult(data);
      addToast('Classification complete', 'success');
      loadStatus(true); // update counts
    } catch (err) {
      addToast('Failed to run classification', 'error');
    } finally {
      setIsClassifying(false);
    }
  };

  // Resolve / Verify a ticket in human queue
  const handleResolveTicket = async (ticketId, correctedLabel) => {
    try {
      await apiFetch(`/api/tickets/${ticketId}/resolve`, {
        method: 'POST',
        body: JSON.stringify({ corrected_label: correctedLabel })
      });
      addToast(`Ticket #${ticketId} resolved successfully`, 'success');
      
      // Update local state by removing resolved ticket
      setTickets(prev => prev.filter(t => t.id !== ticketId));
      loadStatus(true); // update counts
    } catch (err) {
      addToast('Failed to resolve ticket', 'error');
    }
  };

  // Refresh model pointer cache
  const handleRefreshModelCache = async () => {
    setIsRefreshingStatus(true);
    try {
      await apiFetch('/api/refresh', { method: 'POST' });
      addToast('Inference pipeline reloaded successfully', 'success');
      loadStatus();
    } catch (err) {
      addToast('Failed to reload pipeline', 'error');
    } finally {
      setIsRefreshingStatus(false);
    }
  };

  // Setup tabs loading triggers
  const fetchDashboardData = () => {
    loadStatus(true);
    loadLabels();
    if (activeTab === 'review' || activeTab === 'history') {
      loadTickets();
    } else if (activeTab === 'settings') {
      loadSettings();
    }
  };

  useEffect(() => {
    // Initial verification of stored password
    const password = getPassword();
    // Fetch initial status to check if password matches
    apiFetch('/api/status')
      .then((data) => {
        setStatusData(data);
        setIsAuthenticated(true);
        loadLabels();
      })
      .catch((err) => {
        // If password fails or not found, trigger auth gate modal
        if (password) {
          localStorage.removeItem('dashboard_password');
          addToast('Session expired, please log in again', 'warning');
        }
        setIsAuthenticated(false);
      });
  }, []);

  useEffect(() => {
    if (isAuthenticated) {
      fetchDashboardData();
    }
  }, [activeTab, isAuthenticated]);

  // Toggle accordion item
  const toggleTicketExpand = (id) => {
    setExpandedTickets((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  // Filtered tickets (search bar)
  const filteredTickets = tickets.filter(
    (t) =>
      t.subject.toLowerCase().includes(searchQuery.toLowerCase()) ||
      t.body.toLowerCase().includes(searchQuery.toLowerCase()) ||
      t.predicted_label.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="app-container">
      {/* Toast Notifications */}
      <div className="toast-container">
        {toasts.map((t) => (
          <div key={t.id} className={`toast ${t.type}`}>
            {t.type === 'success' ? (
              <CheckCircle2 size={18} className="toast-icon" />
            ) : (
              <AlertCircle size={18} className="toast-icon" />
            )}
            <span>{t.message}</span>
          </div>
        ))}
      </div>

      {/* Admin Password Gate */}
      {!isAuthenticated && (
        <div className="auth-overlay">
          <form className="auth-modal" onSubmit={handleLogin}>
            <div style={{ display: 'flex', justifyContent: 'center' }}>
              <Lock size={40} className="logo-icon" />
            </div>
            <h2 className="auth-title">Dashboard Auth</h2>
            <p className="status-label" style={{ marginTop: '-12px' }}>
              Enter your dashboard admin password to access statistics, parameters, and review logs.
            </p>
            <div className="form-group" style={{ textAlign: 'left' }}>
              <label className="form-label">Admin Password</label>
              <input
                type="password"
                className="form-input"
                placeholder="Enter password..."
                value={passwordInput}
                onChange={(e) => setPasswordInput(e.target.value)}
                required
              />
            </div>
            {authError && (
              <div style={{ color: 'var(--accent-red)', fontSize: '0.85rem', fontWeight: 600 }}>
                {authError}
              </div>
            )}
            <button type="submit" className="btn btn-primary" style={{ width: '100%' }}>
              Unlock Dashboard
            </button>
          </form>
        </div>
      )}

      {/* Sidebar navigation */}
      <aside className="sidebar">
        <div className="logo-section">
          <Layers size={24} className="logo-icon" />
          <span className="logo-text">Classifier Portal</span>
        </div>

        <nav className="nav-links">
          <button
            className={`nav-link ${activeTab === 'test' ? 'active' : ''}`}
            onClick={() => setActiveTab('test')}
          >
            <Layers size={18} className="nav-icon" />
            <span>Test Classifier</span>
          </button>
          <button
            className={`nav-link ${activeTab === 'review' ? 'active' : ''}`}
            onClick={() => setActiveTab('review')}
          >
            <ListTodo size={18} className="nav-icon" />
            <span>Review Queue</span>
          </button>
          <button
            className={`nav-link ${activeTab === 'history' ? 'active' : ''}`}
            onClick={() => setActiveTab('history')}
          >
            <History size={18} className="nav-icon" />
            <span>Processed Logs</span>
          </button>
          <button
            className={`nav-link ${activeTab === 'settings' ? 'active' : ''}`}
            onClick={() => setActiveTab('settings')}
          >
            <Settings size={18} className="nav-icon" />
            <span>Routing Settings</span>
          </button>
        </nav>

        {/* Sidebar CLP widget */}
        <div className="sidebar-status-widget">
          <h3 className="status-widget-title">CLP System</h3>
          <div className="status-widget-item">
            <span className="status-label">Active Model</span>
            <span className="status-value">
              <Sparkles size={13} style={{ color: 'var(--accent-indigo)' }} />
              <code style={{ fontSize: '0.8rem' }}>
                {statusData.active_model && statusData.active_model.substring(0, 16)}
              </code>
            </span>
          </div>

          <div className="status-widget-item">
            <span className="status-label">Retraining Queue</span>
            <span className="status-value">
              <Database size={13} style={{ color: 'var(--accent-cyan)' }} />
              <span>{statusData.feedback_count} / 50 events</span>
            </span>
          </div>

          {statusData.latest_run && (
            <div className="status-widget-item">
              <span className="status-label">Pipeline Status</span>
              <span className="status-value">
                {statusData.latest_run.status === 'in_progress' ? (
                  <span className="badge badge-warning" style={{ padding: '2px 6px', fontSize: '0.65rem' }}>
                    <span className="pulse-dot"></span> Train
                  </span>
                ) : statusData.latest_run.conclusion === 'success' ? (
                  <span className="badge badge-success" style={{ padding: '2px 6px', fontSize: '0.65rem' }}>
                    Stable
                  </span>
                ) : (
                  <span className="badge badge-danger" style={{ padding: '2px 6px', fontSize: '0.65rem' }}>
                    Failed
                  </span>
                )}
              </span>
            </div>
          )}

          <div style={{ marginTop: '16px', display: 'flex', gap: '8px' }}>
            <button
              onClick={() => handleRefreshModelCache()}
              className="btn btn-secondary btn-icon-only"
              style={{ width: '100%', height: '32px' }}
              title="Refresh status"
              disabled={isRefreshingStatus}
            >
              <RefreshCw size={13} className={isRefreshingStatus ? 'spinner' : ''} />
            </button>
            <button
              onClick={handleLogout}
              className="btn btn-secondary btn-icon-only"
              style={{ width: '100%', height: '32px', color: 'var(--accent-red)' }}
              title="Log out"
            >
              <X size={13} />
            </button>
          </div>
        </div>
      </aside>

      {/* Main content body */}
      <main className="main-content">
        <header className="header-bar">
          <div>
            <h1 className="page-title">
              {activeTab === 'test' && 'Test Classifier'}
              {activeTab === 'review' && 'Human Review Queue'}
              {activeTab === 'history' && 'Processed Logs'}
              {activeTab === 'settings' && 'Routing Settings'}
            </h1>
            <p className="status-label" style={{ marginTop: '4px' }}>
              {activeTab === 'test' && 'Evaluate model classifications and route manually in real-time'}
              {activeTab === 'review' && 'Validate low-confidence items and submit corrections to train the next model'}
              {activeTab === 'history' && 'Review resolved emails history and corrections audit log'}
              {activeTab === 'settings' && 'Configure destination routing emails for ticket categories'}
            </p>
          </div>

          <div className="header-actions">
            {statusData.latest_run && (
              <a
                href={statusData.latest_run.html_url}
                target="_blank"
                rel="noopener noreferrer"
                className="btn btn-secondary"
                style={{ fontSize: '0.8rem', padding: '6px 12px' }}
              >
                <span>Pipeline Logs</span>
                <ExternalLink size={12} />
              </a>
            )}
            <button
              onClick={() => fetchDashboardData()}
              className="btn btn-secondary btn-icon-only"
              disabled={isRefreshingStatus}
            >
              <RefreshCw size={16} className={isRefreshingStatus ? 'spinner' : ''} />
            </button>
          </div>
        </header>

        {/* METRICS HEADER */}
        {activeTab !== 'settings' && (
          <section className="metrics-grid">
            <div className="glass-card metric-card">
              <div className="metric-icon-box">
                <Sparkles size={20} />
              </div>
              <div className="metric-content">
                <span className="metric-value" style={{ fontSize: '1.1rem', wordBreak: 'break-all' }}>
                  {statusData.active_model}
                </span>
                <span className="metric-label">Active Model Revision</span>
              </div>
            </div>

            <div className="glass-card metric-card">
              <div className="metric-icon-box cyan">
                <Database size={20} />
              </div>
              <div className="metric-content">
                <span className="metric-value">{statusData.feedback_count}</span>
                <span className="metric-label">Feedback Events (retrains at 50)</span>
              </div>
            </div>

            <div className="glass-card metric-card">
              <div className="metric-icon-box green">
                <CheckCircle2 size={20} />
              </div>
              <div className="metric-content">
                <span className="metric-value">
                  {statusData.latest_run
                    ? statusData.latest_run.status === 'in_progress'
                      ? 'Retraining'
                      : statusData.latest_run.conclusion === 'success'
                      ? 'Success'
                      : 'Failed'
                    : 'Stable'}
                </span>
                <span className="metric-label">Pipeline Status</span>
              </div>
            </div>
          </section>
        )}

        {/* TAB content panels */}

        {/* Tab 1: Test Classifier */}
        {activeTab === 'test' && (
          <div className="glass-card">
            <h2 className="card-title">
              <Layers size={18} className="nav-icon" />
              <span>Evaluate Email Template</span>
            </h2>
            <form onSubmit={handleClassify} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              <div className="form-group">
                <label className="form-label">Subject</label>
                <input
                  type="text"
                  className="form-input"
                  placeholder="e.g. Refund request for order #19203"
                  value={testSubject}
                  onChange={(e) => setTestSubject(e.target.value)}
                />
              </div>

              <div className="form-group">
                <label className="form-label">Email Body</label>
                <textarea
                  className="form-textarea"
                  placeholder="e.g. I am writing to ask for a full refund because my order arrived damaged..."
                  value={testBody}
                  onChange={(e) => setTestBody(e.target.value)}
                ></textarea>
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <button type="submit" className="btn btn-primary" disabled={isClassifying}>
                  {isClassifying ? (
                    <>
                      <div className="spinner"></div>
                      <span>Classifying...</span>
                    </>
                  ) : (
                    <>
                      <Send size={16} />
                      <span>Classify Ticket</span>
                    </>
                  )}
                </button>
              </div>
            </form>

            {classifyResult && (
              <div style={{ marginTop: '32px', paddingTop: '24px', borderTop: '1px solid var(--glass-border)' }}>
                <h3 className="card-title" style={{ fontSize: '1rem', marginBottom: '16px' }}>
                  <TrendingUp size={16} style={{ color: 'var(--accent-indigo)' }} />
                  <span>Model Prediction</span>
                </h3>

                <div className="predict-result-grid">
                  <div className="predict-metric">
                    <span className="status-label">Predicted Label</span>
                    <span className="predict-metric-val">{classifyResult.label}</span>
                  </div>

                  <div className="predict-metric">
                    <span className="status-label">Confidence Score</span>
                    <span
                      className={`predict-metric-val ${
                        classifyResult.confidence >= 0.85
                          ? 'high-conf'
                          : classifyResult.confidence >= 0.6
                          ? 'med-conf'
                          : 'low-conf'
                      }`}
                    >
                      {(classifyResult.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>

                <div
                  className={`badge ${
                    classifyResult.status === 'auto_routed' ? 'badge-success' : 'badge-warning'
                  }`}
                  style={{ marginTop: '20px', padding: '8px 12px' }}
                >
                  <AlertCircle size={14} />
                  <span>{classifyResult.routing}</span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Tab 2: Review Queue */}
        {activeTab === 'review' && (
          <div className="glass-card">
            <div style={{ display: 'flex', justifyItems: 'center', justifyContent: 'space-between', marginBottom: '24px' }}>
              <h2 className="card-title" style={{ margin: 0 }}>
                <ListTodo size={18} className="nav-icon" />
                <span>Pending Queue ({filteredTickets.length})</span>
              </h2>

              <div style={{ position: 'relative', width: '300px' }}>
                <Search
                  size={16}
                  style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }}
                />
                <input
                  type="text"
                  className="form-input"
                  placeholder="Search subject or label..."
                  style={{ paddingLeft: '36px', height: '36px' }}
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>
            </div>

            {isLoadingTickets ? (
              <div className="empty-state" style={{ borderStyle: 'solid' }}>
                <div className="spinner" style={{ width: '30px', height: '30px' }}></div>
                <span className="empty-text">Loading ticket review list...</span>
              </div>
            ) : filteredTickets.length === 0 ? (
              <div className="empty-state">
                <CheckCircle2 size={40} className="empty-icon" style={{ color: 'var(--accent-green)' }} />
                <span className="empty-text">Queue is clear! No pending reviews.</span>
                <span className="status-label">All inbound emails have been classified or resolved.</span>
              </div>
            ) : (
              <div className="tickets-list">
                {filteredTickets.map((t) => {
                  const isExpanded = expandedTickets.has(t.id);
                  return (
                    <div key={t.id} className="ticket-item">
                      <div className="ticket-header" onClick={() => toggleTicketExpand(t.id)}>
                        <div className="ticket-title-section">
                          <span className="ticket-id">#{t.id}</span>
                          <span className="ticket-subject">{t.subject}</span>
                        </div>

                        <div className="ticket-meta-section">
                          <span className={`badge ${t.status === 'auto_routed' ? 'badge-cyan' : 'badge-warning'}`}>
                            {t.status === 'auto_routed' ? 'Routed' : 'Review'}
                          </span>
                          <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                            {(t.confidence * 100).toFixed(0)}%
                          </span>
                          {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                        </div>
                      </div>

                      {isExpanded && (
                        <div className="ticket-body-container">
                          <div>
                            <span className="form-label" style={{ marginBottom: '6px', display: 'block' }}>Email Text</span>
                            <div className="ticket-body-text">{t.body || '[Empty Email Body]'}</div>
                          </div>

                          <div className="ticket-resolution-panel">
                            <div className="form-group" style={{ margin: 0, width: '300px' }}>
                              <label className="form-label">Category Correction</label>
                              <select
                                className="form-select"
                                value={correctedLabels[t.id] || t.predicted_label}
                                onChange={(e) =>
                                  setCorrectedLabels((prev) => ({ ...prev, [t.id]: e.target.value }))
                                }
                              >
                                {labels.map((lbl) => (
                                  <option key={lbl} value={lbl}>
                                    {lbl.replace('_', ' ')}
                                  </option>
                                ))}
                              </select>
                            </div>

                            <div style={{ display: 'flex', gap: '12px' }}>
                              <button
                                className="btn btn-secondary"
                                onClick={() => toggleTicketExpand(t.id)}
                              >
                                Cancel
                              </button>
                              <button
                                className="btn btn-primary"
                                onClick={() => handleResolveTicket(t.id, correctedLabels[t.id])}
                              >
                                <Check size={16} />
                                <span>Validate & Resolve</span>
                              </button>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* Tab 3: Processed Logs */}
        {activeTab === 'history' && (
          <div className="glass-card">
            <div style={{ display: 'flex', justifyItems: 'center', justifyContent: 'space-between', marginBottom: '24px' }}>
              <h2 className="card-title" style={{ margin: 0 }}>
                <History size={18} className="nav-icon" />
                <span>Resolved Tickets ({filteredTickets.length})</span>
              </h2>

              <div style={{ position: 'relative', width: '300px' }}>
                <Search
                  size={16}
                  style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }}
                />
                <input
                  type="text"
                  className="form-input"
                  placeholder="Search processed subject..."
                  style={{ paddingLeft: '36px', height: '36px' }}
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>
            </div>

            {isLoadingTickets ? (
              <div className="empty-state" style={{ borderStyle: 'solid' }}>
                <div className="spinner" style={{ width: '30px', height: '30px' }}></div>
                <span className="empty-text">Loading audit history...</span>
              </div>
            ) : filteredTickets.length === 0 ? (
              <div className="empty-state">
                <AlertCircle size={40} className="empty-icon" />
                <span className="empty-text">No resolved logs yet</span>
                <span className="status-label">Resolved tickets from the queue will appear here as audit records.</span>
              </div>
            ) : (
              <div className="tickets-list">
                {filteredTickets.map((t) => {
                  const isExpanded = expandedTickets.has(t.id);
                  const isCorrected = t.human_label && t.human_label !== t.predicted_label;
                  return (
                    <div key={t.id} className="ticket-item">
                      <div className="ticket-header" onClick={() => toggleTicketExpand(t.id)}>
                        <div className="ticket-title-section">
                          <span className="ticket-id">#{t.id}</span>
                          <span className="ticket-subject">{t.subject}</span>
                        </div>

                        <div className="ticket-meta-section">
                          <span className="status-label" style={{ fontSize: '0.75rem' }}>
                            Predicted: <strong>{t.predicted_label}</strong>
                          </span>
                          {t.human_label && (
                            <span
                              className={`badge ${isCorrected ? 'badge-warning' : 'badge-success'}`}
                              style={{ fontSize: '0.7rem', padding: '2px 8px' }}
                            >
                              {isCorrected ? `Corrected to: ${t.human_label}` : 'Validated'}
                            </span>
                          )}
                          {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                        </div>
                      </div>

                      {isExpanded && (
                        <div className="ticket-body-container">
                          <div>
                            <span className="form-label" style={{ marginBottom: '6px', display: 'block' }}>Email text body</span>
                            <div className="ticket-body-text">{t.body || '[Empty Email Body]'}</div>
                          </div>

                          <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: '1px solid var(--glass-border)', paddingTop: '16px', fontSize: '0.85rem' }}>
                            <span className="status-label">
                              Processed on: <strong>{new Date(t.created_at).toLocaleString()}</strong>
                            </span>
                            <span className="status-label">
                              Original Confidence: <strong>{(t.confidence * 100).toFixed(0)}%</strong>
                            </span>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* Tab 4: Settings */}
        {activeTab === 'settings' && (
          <div className="glass-card">
            <h2 className="card-title">
              <Mail size={18} className="nav-icon" />
              <span>Routing Settings Matrix</span>
            </h2>

            {isLoadingSettings ? (
              <div className="empty-state" style={{ borderStyle: 'solid' }}>
                <div className="spinner" style={{ width: '30px', height: '30px' }}></div>
                <span className="empty-text">Loading settings...</span>
              </div>
            ) : (
              <form onSubmit={saveSettings} style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  {labels.map((lbl) => (
                    <div key={lbl} className="form-group" style={{ margin: 0 }}>
                      <label className="form-label" style={{ textTransform: 'capitalize' }}>
                        📧 Destination Email for "{lbl.replace('_', ' ')}"
                      </label>
                      <input
                        type="email"
                        className="form-input"
                        placeholder={`e.g. ${lbl.replace('_', '')}@company.com`}
                        value={settings[lbl] || ''}
                        onChange={(e) =>
                          setSettings((prev) => ({ ...prev, [lbl]: e.target.value }))
                        }
                      />
                    </div>
                  ))}
                </div>

                <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '16px' }}>
                  <button
                    type="submit"
                    className="btn btn-primary"
                    disabled={isSavingSettings}
                  >
                    {isSavingSettings ? (
                      <>
                        <div className="spinner"></div>
                        <span>Saving...</span>
                      </>
                    ) : (
                      <>
                        <Check size={16} />
                        <span>Save Settings</span>
                      </>
                    )}
                  </button>
                </div>
              </form>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
