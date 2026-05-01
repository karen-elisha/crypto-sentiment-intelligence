// ============================================
// CRYPTO SENTIMENT INTELLIGENCE — DASHBOARD JS
// ============================================

const COLORS = {
    blue: '#6382ff',
    cyan: '#22d3ee',
    green: '#34d399',
    red: '#f87171',
    amber: '#fbbf24',
    purple: '#a78bfa',
    pink: '#f472b6',
    clusters: ['#6382ff', '#f87171', '#34d399', '#fbbf24', '#a78bfa'],
    sentimentMap: {
        'Extreme Fear': '#f87171',
        'Fear': '#fbbf24',
        'Neutral': '#8892a8',
        'Greed': '#34d399',
        'Extreme Greed': '#22d3ee'
    }
};

// Chart.js global defaults
Chart.defaults.color = '#8892a8';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.plugins.legend.labels.boxWidth = 12;
Chart.defaults.plugins.legend.labels.padding = 16;

function formatCurrency(n) {
    if (Math.abs(n) >= 1e6) return '$' + (n/1e6).toFixed(1) + 'M';
    if (Math.abs(n) >= 1e3) return '$' + (n/1e3).toFixed(1) + 'K';
    return '$' + n.toFixed(2);
}
function formatPct(n) { return (n * 100).toFixed(1) + '%'; }

async function loadDashboard() {
    try {
        const resp = await fetch('./dashboard/analysis_results.json');
        if (!resp.ok) throw new Error('analysis_results.json not found');
        const data = await resp.json();
        renderDashboard(data);
    } catch (err) {
        document.getElementById('loading-screen').innerHTML =
            `<div style="text-align:center;color:#f87171;padding:40px;">
                <p style="font-size:1.2rem;font-weight:700;">❌ Error Loading Dashboard</p>
                <p style="color:#8892a8;margin-top:12px;">${err.message}</p>
                <p style="color:#555e73;margin-top:8px;">Run <code>python src/analysis.py</code> first</p>
            </div>`;
        console.error(err);
    }
}

function renderDashboard(data) {
    const ins = data.insights || {};

    // Header badges
    document.getElementById('badge-trades').textContent = (ins.total_trades || 0).toLocaleString() + ' trades';
    document.getElementById('badge-accounts').textContent = (ins.total_accounts || 0) + ' accounts';
    if (ins.date_range) document.getElementById('date-range-text').textContent = ins.date_range;

    // KPIs
    const pnlVal = ins.total_pnl || 0;
    document.getElementById('kpi-pnl-value').textContent = formatCurrency(pnlVal);
    document.getElementById('kpi-pnl-value').className = 'kpi-value ' + (pnlVal >= 0 ? 'kpi-positive' : 'kpi-negative');
    document.getElementById('kpi-pnl-sub').textContent = `across ${(ins.total_trades || 0).toLocaleString()} trades`;

    document.getElementById('kpi-wr-value').textContent = formatPct(ins.overall_win_rate || 0);

    document.getElementById('kpi-sent-value').textContent = ins.best_sentiment || '—';
    document.getElementById('kpi-sent-sub').textContent = `avg PnL: ${formatCurrency(ins.best_sentiment_pnl || 0)}`;

    document.getElementById('kpi-model-value').textContent = ins.best_model || '—';
    document.getElementById('kpi-model-sub').textContent = `${formatPct(ins.best_model_accuracy || 0)} accuracy`;

    // Insight Banner
    document.getElementById('insight-text').innerHTML =
        `Trading during <span class="insight-highlight">${ins.best_sentiment}</span> sentiment yields the highest avg PnL of ` +
        `<span class="insight-highlight">${formatCurrency(ins.best_sentiment_pnl || 0)}</span> per trade. ` +
        `The <span class="insight-highlight">${ins.best_strategy}</span> strategy outperforms, and the best trading hour is ` +
        `<span class="insight-highlight">${ins.best_hour || 0}:00</span> (avg PnL: ${formatCurrency(ins.best_hour_pnl || 0)}).`;

    // Time Series Chart
    renderTimeSeries(data.timeseries);

    // Sentiment
    renderSentiment(data.sentiment_perf);

    // Strategies
    renderStrategies(data.strategies);

    // Models
    renderModels(data.models, data.best_model);

    // Feature Importance
    renderFeatureImportance(data.feature_importance);

    // Confusion Matrix
    renderConfusionMatrix(data.confusion_matrix);

    // Clustering
    renderClusters(data.clustering, data.cluster_profiles);

    // Coins
    renderCoins(data.coin_performance);

    // Hourly
    renderHourly(data.hourly);

    // Top Accounts
    renderTopAccounts(data.top_accounts);

    // Correlation
    renderCorrelation(data.correlations);

    // Show dashboard
    document.getElementById('loading-screen').classList.add('fade-out');
    document.getElementById('dashboard').style.display = 'block';
    setTimeout(() => { document.getElementById('loading-screen').style.display = 'none'; }, 600);
}

// ========================
// TIME SERIES
// ========================
function renderTimeSeries(ts) {
    if (!ts || !ts.dates) return;
    const ctx = document.getElementById('pnlChart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: ts.dates,
            datasets: [
                {
                    label: 'Cumulative PnL',
                    data: ts.cumulative_pnl,
                    borderColor: COLORS.blue,
                    backgroundColor: 'rgba(99,130,255,0.06)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0,
                    borderWidth: 2,
                    yAxisID: 'y'
                },
                {
                    label: 'Sentiment Index',
                    data: ts.sentiment,
                    borderColor: COLORS.amber,
                    backgroundColor: 'transparent',
                    borderWidth: 1.5,
                    borderDash: [4,4],
                    pointRadius: 0,
                    tension: 0.3,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { position: 'top' },
                tooltip: {
                    backgroundColor: 'rgba(10,14,26,0.95)',
                    borderColor: 'rgba(99,130,255,0.2)',
                    borderWidth: 1,
                    padding: 12,
                    titleFont: { weight: '600' }
                }
            },
            scales: {
                x: { ticks: { maxTicksLimit: 12, font: { size: 10 } } },
                y: {
                    position: 'left',
                    title: { display: true, text: 'Cumulative PnL ($)', font: { size: 11 } },
                    ticks: { callback: v => formatCurrency(v) }
                },
                y1: {
                    position: 'right',
                    title: { display: true, text: 'Sentiment', font: { size: 11 } },
                    min: 0, max: 100,
                    grid: { drawOnChartArea: false }
                }
            }
        }
    });
}

// ========================
// SENTIMENT
// ========================
function renderSentiment(sentPerf) {
    if (!sentPerf || !sentPerf.length) return;
    const order = ['Extreme Greed', 'Greed', 'Neutral', 'Fear', 'Extreme Fear'];
    const sorted = sentPerf.sort((a,b) => order.indexOf(a.classification) - order.indexOf(b.classification));
    const maxPnl = Math.max(...sorted.map(s => Math.abs(s.avg_pnl)));

    // Visual bars
    const barsEl = document.getElementById('sentiment-bars');
    barsEl.innerHTML = sorted.map(s => {
        const pct = Math.abs(s.avg_pnl) / maxPnl * 100;
        const color = COLORS.sentimentMap[s.classification] || COLORS.blue;
        return `<div class="sent-bar-row">
            <span class="sent-bar-label">${s.classification}</span>
            <div class="sent-bar-track">
                <div class="sent-bar-fill" style="width:${pct}%;background:${color};">
                    ${formatCurrency(s.avg_pnl)}
                </div>
            </div>
        </div>`;
    }).join('');

    // Table
    const tbody = document.querySelector('#sentiment-table tbody');
    tbody.innerHTML = sorted.map(s => `<tr>
        <td><span style="color:${COLORS.sentimentMap[s.classification] || '#fff'};font-weight:600;">${s.classification}</span></td>
        <td style="color:${s.avg_pnl >= 0 ? COLORS.green : COLORS.red};font-weight:700;">${formatCurrency(s.avg_pnl)}</td>
        <td>${formatPct(s.win_rate)}</td>
        <td>${(s.trade_count || 0).toLocaleString()}</td>
        <td>${(s.sharpe_like || 0).toFixed(4)}</td>
    </tr>`).join('');
}

// ========================
// STRATEGIES
// ========================
function renderStrategies(strategies) {
    if (!strategies || !strategies.length) return;
    const bestPnl = Math.max(...strategies.map(s => s.avg_pnl || 0));
    const content = document.getElementById('strategy-content');

    content.innerHTML = `<div class="strat-cards">${strategies.map(s => {
        const isWin = s.avg_pnl === bestPnl;
        return `<div class="strat-card ${isWin ? 'winner' : ''}">
            <div class="strat-name">${isWin ? '👑 ' : ''}${s.strategy}</div>
            <div class="strat-pnl" style="color:${s.avg_pnl >= 0 ? COLORS.green : COLORS.red};">
                ${formatCurrency(s.avg_pnl || 0)}
            </div>
            <div class="strat-detail">Win Rate: ${formatPct(s.win_rate || 0)} • ${(s.trades || 0).toLocaleString()} trades</div>
            <div class="strat-detail">Total PnL: ${formatCurrency(s.total_pnl || 0)}</div>
        </div>`;
    }).join('')}</div>`;

    // Chart
    const ctx = document.getElementById('strategyChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: strategies.map(s => s.strategy),
            datasets: [
                {
                    label: 'Avg PnL ($)',
                    data: strategies.map(s => s.avg_pnl || 0),
                    backgroundColor: strategies.map(s => s.avg_pnl >= 0 ? 'rgba(52,211,153,0.7)' : 'rgba(248,113,113,0.7)'),
                    borderColor: strategies.map(s => s.avg_pnl >= 0 ? COLORS.green : COLORS.red),
                    borderWidth: 1,
                    barPercentage: 0.6,
                    categoryPercentage: 0.7
                },
                {
                    label: 'Win Rate (%)',
                    data: strategies.map(s => (s.win_rate || 0) * 100),
                    backgroundColor: 'rgba(99,130,255,0.6)',
                    borderColor: COLORS.blue,
                    borderWidth: 1,
                    barPercentage: 0.6,
                    categoryPercentage: 0.7
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'top' } },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.04)' } },
                y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.04)' } }
            }
        }
    });
}

// ========================
// MODELS
// ========================
function renderModels(models, bestName) {
    if (!models || !models.length) return;
    const container = document.getElementById('model-comparison');
    container.innerHTML = `<div class="model-cards">${models.map(m => {
        const isBest = m.name === bestName;
        return `<div class="model-row ${isBest ? 'best' : ''}">
            <span class="model-name">${isBest ? '🏆 ' : ''}${m.name}</span>
            ${isBest ? '<span class="model-badge">BEST</span>' : ''}
            <span class="model-metric">Acc: <strong>${formatPct(m.accuracy)}</strong></span>
            <span class="model-metric">F1: <strong>${m.f1_score?.toFixed(3) || '—'}</strong></span>
            <span class="model-metric">AUC: <strong>${m.auc_roc?.toFixed(3) || '—'}</strong></span>
            <span class="model-metric">CV: <strong>${formatPct(m.cv_mean)}±${formatPct(m.cv_std)}</strong></span>
        </div>`;
    }).join('')}</div>`;

    const ctx = document.getElementById('modelChart').getContext('2d');
    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC', 'CV Mean'],
            datasets: models.map((m, i) => ({
                label: m.name,
                data: [m.accuracy, m.precision, m.recall, m.f1_score, m.auc_roc || 0, m.cv_mean],
                borderColor: COLORS.clusters[i],
                backgroundColor: COLORS.clusters[i] + '15',
                borderWidth: 2,
                pointRadius: 3
            }))
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: { r: { min: 0, max: 1, ticks: { stepSize: 0.2, font: { size: 9 } }, grid: { color: 'rgba(255,255,255,0.06)' } } },
            plugins: { legend: { position: 'top' } }
        }
    });
}

// ========================
// FEATURE IMPORTANCE
// ========================
function renderFeatureImportance(features) {
    if (!features || !features.length) return;
    const ctx = document.getElementById('featureChart').getContext('2d');
    const sorted = features.sort((a,b) => b.importance - a.importance);
    const featureColors = [COLORS.cyan, COLORS.blue, COLORS.green, COLORS.amber, COLORS.purple, COLORS.pink, COLORS.red, '#64748b', '#94a3b8'];
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: sorted.map(f => f.feature),
            datasets: [{
                label: 'Importance',
                data: sorted.map(f => f.importance),
                backgroundColor: sorted.map((_, i) => featureColors[i % featureColors.length]),
                borderColor: sorted.map((_, i) => featureColors[i % featureColors.length]),
                borderWidth: 1,
                barPercentage: 0.7,
                categoryPercentage: 0.8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: { legend: { display: false } },
            scales: {
                x: { beginAtZero: true, title: { display: true, text: 'Importance' }, grid: { color: 'rgba(255,255,255,0.04)' } },
                y: { ticks: { font: { size: 11, weight: '500' } }, grid: { color: 'rgba(255,255,255,0.04)' } }
            }
        }
    });
}

// ========================
// CONFUSION MATRIX
// ========================
function renderConfusionMatrix(cm) {
    if (!cm) return;
    const el = document.getElementById('confusion-matrix');
    const total = cm.true_pos + cm.true_neg + cm.false_pos + cm.false_neg;
    el.innerHTML = `
        <h3 class="section-title">Confusion Matrix</h3>
        <div class="cm-grid">
            <div class="cm-header"></div>
            <div class="cm-header">Pred Loss</div>
            <div class="cm-header">Pred Profit</div>
            <div class="cm-label">Actual Loss</div>
            <div class="cm-cell cm-tn">${cm.true_neg.toLocaleString()}<br><span style="font-size:0.65rem;opacity:0.7">${(cm.true_neg/total*100).toFixed(1)}%</span></div>
            <div class="cm-cell cm-fp">${cm.false_pos.toLocaleString()}<br><span style="font-size:0.65rem;opacity:0.7">${(cm.false_pos/total*100).toFixed(1)}%</span></div>
            <div class="cm-label">Actual Profit</div>
            <div class="cm-cell cm-fn">${cm.false_neg.toLocaleString()}<br><span style="font-size:0.65rem;opacity:0.7">${(cm.false_neg/total*100).toFixed(1)}%</span></div>
            <div class="cm-cell cm-tp">${cm.true_pos.toLocaleString()}<br><span style="font-size:0.65rem;opacity:0.7">${(cm.true_pos/total*100).toFixed(1)}%</span></div>
        </div>`;
}

// ========================
// CLUSTERING
// ========================
function renderClusters(clustering, profiles) {
    if (!clustering || !clustering.length) return;
    const ctx = document.getElementById('clusterChart').getContext('2d');

    const groups = {};
    clustering.forEach(p => {
        if (!groups[p.cluster]) groups[p.cluster] = [];
        groups[p.cluster].push({ x: p.x, y: p.y, account: p.Account });
    });

    new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: Object.entries(groups).map(([cl, pts]) => ({
                label: `Cluster ${cl}`,
                data: pts.map(p => ({ x: p.x, y: p.y })),
                backgroundColor: COLORS.clusters[cl] + 'aa',
                borderColor: COLORS.clusters[cl],
                borderWidth: 1,
                pointRadius: 8,
                pointHoverRadius: 12
            }))
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: 'PC1' } },
                y: { title: { display: true, text: 'PC2' } }
            },
            plugins: {
                tooltip: {
                    backgroundColor: 'rgba(10,14,26,0.95)',
                    callbacks: {
                        label: (ctx) => {
                            const match = clustering.find(p => Math.abs(p.x - ctx.raw.x) < 0.001 && Math.abs(p.y - ctx.raw.y) < 0.001);
                            if (!match) return '';
                            return [`Account: ${match.Account.substring(0,12)}...`, `PnL: ${formatCurrency(match.total_pnl)}`, `Win Rate: ${formatPct(match.win_rate)}`];
                        }
                    }
                }
            }
        }
    });

    // Cluster profiles
    if (profiles && profiles.length) {
        const labels = ['Conservative', 'Aggressive', 'Balanced', 'Niche', 'Elite'];
        document.getElementById('cluster-profiles').innerHTML =
            `<div class="cluster-profile-row">${profiles.map((p, i) => `
                <div class="cluster-badge" style="border-color:${COLORS.clusters[i]}33;">
                    <div class="cluster-badge-title" style="color:${COLORS.clusters[i]};">Cluster ${p.cluster} · ${labels[i]}</div>
                    <div class="cluster-badge-stat">${p.count} accounts</div>
                    <div class="cluster-badge-stat">Avg PnL: ${formatCurrency(p.avg_pnl)}</div>
                    <div class="cluster-badge-stat">Win Rate: ${formatPct(p.avg_win_rate)}</div>
                </div>`).join('')}</div>`;
    }
}

// ========================
// COINS
// ========================
function renderCoins(coins) {
    if (!coins || !coins.length) return;
    const ctx = document.getElementById('coinChart').getContext('2d');
    const top = coins.slice(0, 12);
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: top.map(c => c.Coin),
            datasets: [{
                label: 'Total PnL',
                data: top.map(c => c.total_pnl),
                backgroundColor: top.map(c => c.total_pnl >= 0 ? 'rgba(52,211,153,0.65)' : 'rgba(248,113,113,0.65)'),
                borderColor: top.map(c => c.total_pnl >= 0 ? COLORS.green : COLORS.red),
                borderWidth: 1,
                barPercentage: 0.7,
                categoryPercentage: 0.8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.04)' } },
                y: { beginAtZero: true, ticks: { callback: v => formatCurrency(v) }, grid: { color: 'rgba(255,255,255,0.04)' } }
            }
        }
    });
}

// ========================
// HOURLY
// ========================
function renderHourly(hourly) {
    if (!hourly || !hourly.length) return;
    const ctx = document.getElementById('hourlyChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: hourly.map(h => `${h.hour}:00`),
            datasets: [
                {
                    label: 'Avg PnL',
                    data: hourly.map(h => h.avg_pnl),
                    backgroundColor: hourly.map(h => h.avg_pnl >= 0 ? 'rgba(52,211,153,0.5)' : 'rgba(248,113,113,0.5)'),
                    borderRadius: 4,
                    yAxisID: 'y'
                },
                {
                    label: 'Win Rate',
                    data: hourly.map(h => (h.win_rate || 0) * 100),
                    type: 'line',
                    borderColor: COLORS.cyan,
                    borderWidth: 2,
                    pointRadius: 3,
                    tension: 0.4,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { position: 'left', title: { display: true, text: 'Avg PnL ($)' } },
                y1: { position: 'right', title: { display: true, text: 'Win Rate (%)' }, min: 0, max: 100, grid: { drawOnChartArea: false } }
            }
        }
    });
}

// ========================
// TOP ACCOUNTS
// ========================
function renderTopAccounts(accounts) {
    if (!accounts || !accounts.length) return;
    const tbody = document.querySelector('#top-accounts-table tbody');
    tbody.innerHTML = accounts.map((a, i) => `<tr>
        <td><span style="color:${COLORS.clusters[i % 5]};font-weight:600;">${['🥇','🥈','🥉','4️⃣','5️⃣'][i]} ${a.Account}</span></td>
        <td style="color:${COLORS.green};font-weight:700;">${formatCurrency(a.total_pnl)}</td>
        <td>${formatPct(a.win_rate)}</td>
        <td>${(a.trade_count || 0).toLocaleString()}</td>
        <td>${(a.sharpe || 0).toFixed(4)}</td>
    </tr>`).join('');
}

// ========================
// CORRELATION
// ========================
function renderCorrelation(corr) {
    if (!corr || !corr.matrix) return;
    const ctx = document.getElementById('corrChart').getContext('2d');
    const labels = corr.labels;
    const matrix = corr.matrix;

    // Build scatter-like heatmap data
    const dataPoints = [];
    matrix.forEach((row, i) => {
        row.forEach((val, j) => {
            dataPoints.push({ x: j, y: i, v: val });
        });
    });

    // Use a bubble chart as heatmap
    new Chart(ctx, {
        type: 'bubble',
        data: {
            datasets: [{
                data: dataPoints.map(d => ({
                    x: d.x, y: d.y, r: Math.abs(d.v) * 18
                })),
                backgroundColor: dataPoints.map(d =>
                    d.v > 0 ? `rgba(52,211,153,${Math.abs(d.v) * 0.7})` : `rgba(248,113,113,${Math.abs(d.v) * 0.7})`
                )
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            const d = dataPoints[ctx.dataIndex];
                            return `${labels[d.y]} × ${labels[d.x]}: ${d.v.toFixed(3)}`;
                        }
                    }
                }
            },
            scales: {
                x: { min: -0.5, max: labels.length - 0.5, ticks: { callback: (v) => labels[v] || '' } },
                y: { min: -0.5, max: labels.length - 0.5, ticks: { callback: (v) => labels[v] || '' }, reverse: true }
            }
        }
    });
}

// ========================
// BOOT
// ========================
loadDashboard();
