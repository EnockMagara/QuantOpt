/**
 * MMS Finance Web Application - Frontend JavaScript
 */

// Global variables
let selectedTickers = [];
let currentWeights = {};
let currentBacktestData = null;
let weightsChart = null;
let backtestChart = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    loadTickerSuggestions();
    setupEventListeners();
    updateMaxPositionDisplay();
}

function setupEventListeners() {
    // Form submission
    document.getElementById('optimizerForm').addEventListener('submit', handleOptimize);
    
    // Ticker input
    document.getElementById('tickerInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            e.preventDefault();
            addTickerFromInput();
        }
    });
    
    // Parameter sliders
    document.getElementById('maxPosition').addEventListener('input', updateMaxPositionDisplay);
    document.getElementById('riskFreeRate').addEventListener('input', updateRiskFreeRateDisplay);
    document.getElementById('transactionCosts').addEventListener('input', updateTransactionCostsDisplay);
    document.getElementById('minWeight').addEventListener('input', updateMinWeightDisplay);
    document.getElementById('lookbackPeriod').addEventListener('input', updateLookbackPeriodDisplay);
    
    // Backtest button
    document.getElementById('backtestBtn').addEventListener('click', runBacktest);
}

function loadTickerSuggestions() {
    fetch('/api/suggestions')
        .then(response => response.json())
        .then(data => {
            populateSuggestions('stockSuggestions', data.stocks);
            populateSuggestions('etfSuggestions', data.etfs);
            populateSuggestions('cryptoSuggestions', data.crypto);
        })
        .catch(error => {
            console.error('Error loading suggestions:', error);
        });
}

function populateSuggestions(containerId, tickers) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    
    tickers.forEach(ticker => {
        const tag = document.createElement('span');
        tag.className = 'ticker-tag';
        tag.textContent = ticker;
        tag.onclick = () => toggleTicker(ticker);
        container.appendChild(tag);
    });
}

function toggleTicker(ticker) {
    const index = selectedTickers.indexOf(ticker);
    if (index === -1) {
        selectedTickers.push(ticker);
    } else {
        selectedTickers.splice(index, 1);
    }
    updateSelectedTickersDisplay();
}

function addTickerFromInput() {
    const input = document.getElementById('tickerInput');
    const tickers = input.value.split(',').map(t => t.trim().toUpperCase()).filter(t => t);
    
    tickers.forEach(ticker => {
        if (!selectedTickers.includes(ticker)) {
            selectedTickers.push(ticker);
        }
    });
    
    input.value = '';
    updateSelectedTickersDisplay();
}

function updateSelectedTickersDisplay() {
    const container = document.getElementById('selectedTickers');
    container.innerHTML = '';
    
    if (selectedTickers.length > 0) {
        const label = document.createElement('strong');
        label.textContent = 'Selected Tickers: ';
        container.appendChild(label);
        
        selectedTickers.forEach(ticker => {
            const tag = document.createElement('span');
            tag.className = 'ticker-tag selected';
            tag.textContent = ticker;
            tag.onclick = () => toggleTicker(ticker);
            container.appendChild(tag);
        });
    }
}

function updateMaxPositionDisplay() {
    const slider = document.getElementById('maxPosition');
    const display = document.getElementById('maxPositionValue');
    display.textContent = Math.round(slider.value * 100) + '%';
}

function updateRiskFreeRateDisplay() {
    const slider = document.getElementById('riskFreeRate');
    const display = document.getElementById('riskFreeRateValue');
    display.textContent = (slider.value * 100).toFixed(1) + '%';
}

function updateTransactionCostsDisplay() {
    const slider = document.getElementById('transactionCosts');
    const display = document.getElementById('transactionCostsValue');
    display.textContent = (slider.value * 100).toFixed(1) + '%';
}

function updateMinWeightDisplay() {
    const slider = document.getElementById('minWeight');
    const display = document.getElementById('minWeightValue');
    display.textContent = (slider.value * 100).toFixed(1) + '%';
}

function updateLookbackPeriodDisplay() {
    const slider = document.getElementById('lookbackPeriod');
    const display = document.getElementById('lookbackPeriodValue');
    display.textContent = Math.round(slider.value) + ' days';
}

function handleOptimize(e) {
    e.preventDefault();
    
    if (selectedTickers.length === 0) {
        alert('Please select at least one ticker symbol.');
        return;
    }
    
    const formData = {
        tickers: selectedTickers,
        start_date: document.getElementById('startDate').value,
        end_date: document.getElementById('endDate').value,
        risk_tolerance: document.querySelector('input[name="riskTolerance"]:checked').value,
        max_position: parseFloat(document.getElementById('maxPosition').value),
        risk_free_rate: parseFloat(document.getElementById('riskFreeRate').value),
        optimization_method: document.getElementById('optimizationMethod').value,
        rebalancing_frequency: document.getElementById('rebalancingFreq').value,
        transaction_costs: parseFloat(document.getElementById('transactionCosts').value),
        min_weight: parseFloat(document.getElementById('minWeight').value),
        lookback_period: parseInt(document.getElementById('lookbackPeriod').value)
    };
    
    showLoading(true);
    
    fetch('/api/optimize', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
    })
    .then(response => response.json())
    .then(data => {
        showLoading(false);
        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'Optimization failed');
        }
    })
    .catch(error => {
        showLoading(false);
        showError('Network error: ' + error.message);
    });
}

function displayResults(data) {
    currentWeights = data.portfolio_weights;
    
    // Update metrics
    document.getElementById('expectedReturn').textContent = 
        (data.portfolio_metrics.expected_return * 100).toFixed(2) + '%';
    document.getElementById('volatility').textContent = 
        (data.portfolio_metrics.volatility * 100).toFixed(2) + '%';
    document.getElementById('sharpeRatio').textContent = 
        data.portfolio_metrics.sharpe_ratio.toFixed(3);
    
    // Display optimization parameters
    displayOptimizationParams(data.data_summary);
    
    // Create weights chart
    createWeightsChart(data.portfolio_weights);
    
    // Show results section
    document.getElementById('resultsSection').style.display = 'block';
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
}

function displayOptimizationParams(params) {
    const container = document.getElementById('optimizationParams');
    container.innerHTML = '';
    
    const paramCards = [
        { label: 'Optimization Method', value: params.optimization_method.replace('_', ' ').toUpperCase() },
        { label: 'Max Position Size', value: (params.max_position * 100).toFixed(1) + '%' },
        { label: 'Risk-Free Rate', value: (params.risk_free_rate * 100).toFixed(1) + '%' },
        { label: 'Min Weight Threshold', value: (params.min_weight * 100).toFixed(1) + '%' },
        { label: 'Rebalancing Frequency', value: params.rebalancing_frequency.toUpperCase() },
        { label: 'Transaction Costs', value: (params.transaction_costs * 100).toFixed(2) + '%' },
        { label: 'Lookback Period', value: params.lookback_period + ' days' },
        { label: 'Risk Tolerance', value: params.risk_tolerance.toUpperCase() }
    ];
    
    paramCards.forEach(param => {
        const col = document.createElement('div');
        col.className = 'col-md-3 col-sm-6 mb-3';
        col.innerHTML = `
            <div class="card text-center p-2">
                <h6 class="text-muted mb-1">${param.label}</h6>
                <h5 class="mb-0">${param.value}</h5>
            </div>
        `;
        container.appendChild(col);
    });
}

function createWeightsChart(weights) {
    const ctx = document.getElementById('weightsChart').getContext('2d');
    
    if (weightsChart) {
        weightsChart.destroy();
    }
    
    const labels = Object.keys(weights);
    const values = Object.values(weights).map(w => w * 100);
    
    weightsChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: [
                    '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF',
                    '#FF9F40', '#FF6384', '#C9CBCF', '#4BC0C0', '#FF6384'
                ],
                borderWidth: 2,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.label + ': ' + context.parsed.toFixed(2) + '%';
                        }
                    }
                }
            }
        }
    });
}

function runBacktest() {
    if (Object.keys(currentWeights).length === 0) {
        alert('Please optimize a portfolio first.');
        return;
    }
    
    const formData = {
        tickers: selectedTickers,
        weights: currentWeights,
        start_date: document.getElementById('startDate').value,
        end_date: document.getElementById('endDate').value
    };
    
    showLoading(true);
    
    fetch('/api/backtest', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
    })
    .then(response => response.json())
    .then(data => {
        showLoading(false);
        if (data.success) {
            displayBacktestResults(data);
        } else {
            showError(data.error || 'Backtest failed');
        }
    })
    .catch(error => {
        showLoading(false);
        showError('Network error: ' + error.message);
    });
}

function displayBacktestResults(data) {
    currentBacktestData = data.backtest_data;
    
    // Update metrics
    document.getElementById('maxDrawdown').textContent = 
        (data.metrics.max_drawdown * 100).toFixed(2) + '%';
    
    // Create backtest chart
    createBacktestChart(data.backtest_data);
}

function createBacktestChart(backtestData) {
    const ctx = document.getElementById('backtestChart').getContext('2d');
    
    if (backtestChart) {
        backtestChart.destroy();
    }
    
    const dates = backtestData.map(d => d.date);
    const cumulativeReturns = backtestData.map(d => d.cumulative_return * 100);
    
    backtestChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [{
                label: 'Cumulative Return (%)',
                data: cumulativeReturns,
                borderColor: '#36A2EB',
                backgroundColor: 'rgba(54, 162, 235, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'time',
                    time: {
                        parser: 'YYYY-MM-DD',
                        displayFormats: {
                            day: 'MMM DD'
                        }
                    },
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Cumulative Return (%)'
                    }
                }
            },
            plugins: {
                legend: {
                    display: true
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            return 'Return: ' + context.parsed.y.toFixed(2) + '%';
                        }
                    }
                }
            }
        }
    });
}

function showLoading(show) {
    document.getElementById('loading').style.display = show ? 'block' : 'none';
}

function showError(message) {
    alert('Error: ' + message);
}

function scrollToOptimizer() {
    document.getElementById('optimizerForm').scrollIntoView({ behavior: 'smooth' });
}

// Utility functions
function formatNumber(num, decimals = 2) {
    return parseFloat(num).toFixed(decimals);
}

function formatPercentage(num, decimals = 2) {
    return (num * 100).toFixed(decimals) + '%';
}
