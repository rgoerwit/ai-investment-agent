function renderRefresh() {
  const freshness = state.snapshot.freshness;
  const screening = state.snapshot.screening_freshness || {};
  const staleEligible =
    freshness.blocking_now.length + freshness.stale_in_queue.length;
  const dueSoonEligible = freshness.due_soon.length;
  const allFresh =
    staleEligible === 0 &&
    dueSoonEligible === 0 &&
    freshness.candidate_blocked.length === 0 &&
    freshness.fresh_count > 0;
  const explainer = allFresh
    ? "Reload Data in the top bar only rereads the current dashboard data. The controls here queue background analysis reruns if you want to refresh specific tickers anyway."
    : "Reload Data in the top bar rereads the current dashboard data. The controls here queue background analysis reruns; finished jobs show up after the next data reload.";
  return `
    <section>
      <h3 class="section-title">Last Broad Screening Run</h3>
      <div class="topbar-actions" style="justify-content: flex-start; margin-bottom: 0.75rem;">
        <span class="status-pill">${
          screening.status === "missing"
            ? "Missing"
            : screening.status === "stale"
              ? "Overdue"
              : "Fresh"
        }</span>
      </div>
      ${renderCards([
        {
          label: "Screening date",
          value: screening.screening_date || "—",
        },
        {
          label: "Age (days)",
          value:
            screening.age_days === null || screening.age_days === undefined
              ? "—"
              : screening.age_days,
        },
        {
          label: "Candidates screened",
          value:
            screening.candidate_count === null ||
            screening.candidate_count === undefined
              ? "—"
              : screening.candidate_count,
        },
        {
          label: "BUYs found",
          value:
            screening.buy_count === null || screening.buy_count === undefined
              ? "—"
              : screening.buy_count,
        },
      ])}
      <p class="muted">${
        screening.status === "missing"
          ? "No completed broad screening sweep is recorded yet. Run ./scripts/run_pipeline.sh when you want new candidate discovery."
          : screening.status === "stale"
            ? "Broad candidate discovery looks overdue even if per-ticker analyses are fresh."
            : "Broad candidate discovery has run recently."
      }</p>
    </section>
    ${renderCards([
      { label: "Needs review", value: freshness.blocking_now.length },
      { label: "Refresh queue", value: freshness.stale_in_queue.length },
      { label: "Needs full refresh", value: freshness.candidate_blocked.length },
      { label: "Due soon", value: freshness.due_soon.length },
      { label: "Fresh", value: freshness.fresh_count },
    ])}
    <section>
      <h3 class="section-title">Queue Analysis Refresh Job</h3>
      <p class="muted">${escapeHtml(explainer)}</p>
      <div class="jobs-controls">
        <button id="job-stale" type="button" ${staleEligible === 0 ? "disabled" : ""}>Queue action-required reruns (${staleEligible})</button>
        <button id="job-due-soon" type="button" ${dueSoonEligible === 0 ? "disabled" : ""}>Queue due-soon reruns (${dueSoonEligible})</button>
        <input id="job-ticker-input" type="text" placeholder="7203.T, MEGP.L">
        <button id="job-custom" type="button">Queue ticker rerun list</button>
      </div>
    </section>
    <section>
      <h3 class="section-title">Background Analysis Jobs</h3>
      ${renderJobsTable()}
    </section>
  `;
}

function renderJobsTable() {
  if (!state.jobs.length) {
    return "<p class='muted'>No background analysis jobs yet. Queue one above, then use Reload Data after it finishes if you want to see the updated view.</p>";
  }
  const rows = state.jobs
    .map(
      (job) => `
      <tr>
        <td title="${escapeHtmlAttr(job.job_id)}">${escapeHtml((job.job_id || "").slice(0, 8) || "—")}</td>
        <td>${escapeHtml(job.scope)}</td>
        <td>${escapeHtml(job.status)}</td>
        <td>${escapeHtml(job.created_at)}</td>
        <td>${escapeHtml(job.finished_at || "—")}</td>
      </tr>
    `,
    )
    .join("");
  return `<table><thead><tr><th>Job</th><th>Scope</th><th>Status</th><th>Created</th><th>Finished</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function renderSettings() {
  const settings = state.settings || {};
  const modeValue = settings.read_only ? "true" : "false";
  return `
    <section>
      <h3 class="section-title">Dashboard Settings</h3>
      <p class="muted">These settings control the next data load. Use startup flags when you want a one-off session override.</p>
      <form id="settings-form" class="settings-form">
        <label>IBKR account ID<input name="account_id" value="${escapeHtmlAttr(settings.account_id || "")}" placeholder="U1234567"></label>
        <label>Watchlist name<input name="watchlist_name" value="${escapeHtmlAttr(settings.watchlist_name || "")}"></label>
        <label>Data source
          <select name="read_only">
            <option value="false" ${modeValue === "false" ? "selected" : ""}>Live IBKR portfolio</option>
            <option value="true" ${modeValue === "true" ? "selected" : ""}>Read-only results only</option>
          </select>
        </label>
        <!-- 14 / 10 are last-resort fallbacks only used if /api/settings omits a value;
             /api/settings (DashboardSettings ← src/ibkr/portfolio_defaults) is authoritative. -->
        <label>Max age days<input name="max_age_days" type="number" value="${escapeHtmlAttr(settings.max_age_days ?? 14)}"></label>
        <label>Refresh limit<input name="refresh_limit" type="number" value="${escapeHtmlAttr(settings.refresh_limit ?? 10)}"></label>
        <label>Quick mode default
          <select name="quick_mode_default">
            <option value="true" ${settings.quick_mode_default ? "selected" : ""}>true</option>
            <option value="false" ${settings.quick_mode_default === false ? "selected" : ""}>false</option>
          </select>
        </label>
        <label>Notes<textarea name="notes">${escapeHtml(settings.notes || "")}</textarea></label>
        <button type="submit">Save settings</button>
      </form>
    </section>
  `;
}
