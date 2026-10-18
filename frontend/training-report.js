const container = document.querySelector("#report-content");

function addText(parent, tag, value, className) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  element.textContent = value;
  parent.append(element);
  return element;
}

function formatPercent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function renderReport(report) {
  const summary = document.createElement("section");
  summary.className = "report-card";
  addText(summary, "h2", "Evaluation setup");
  const meta = document.createElement("div");
  meta.className = "report-meta";
  [
    ["Dataset", report.dataset],
    ["Rows", report.rows],
    ["Input features", report.features],
    ["Target", report.target],
    ["Training / test", `${report.split.train_rows} / ${report.split.test_rows}`],
    ["Best Extra Trees CV accuracy", formatPercent(report.best_cv_accuracy)],
  ].forEach(([label, value]) => {
    const item = document.createElement("div");
    item.className = "meta-item";
    addText(item, "span", label);
    addText(item, "strong", String(value));
    meta.append(item);
  });
  summary.append(meta);
  const classCounts = Object.entries(report.target_counts)
    .map(([label, count]) => `${label}: ${count}`)
    .join(" · ");
  addText(summary, "p", `Target class counts — ${classCounts}`, "report-note");
  container.append(summary);

  const scoreSection = document.createElement("section");
  scoreSection.className = "report-card";
  addText(scoreSection, "h2", "Model comparison on the held-out test set");
  addText(scoreSection, "p", "Accuracy is the fraction classified correctly. Precision and recall describe positive-class predictions; F1 balances precision and recall. ROC-AUC measures ranking across thresholds.", "report-note");
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"].forEach((name) => addText(headerRow, "th", name));
  thead.append(headerRow);
  const tbody = document.createElement("tbody");
  Object.entries(report.models).forEach(([name, scores]) => {
    const row = document.createElement("tr");
    addText(row, "td", name);
    ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"].forEach((key) => {
      addText(row, "td", formatPercent(scores[key]));
    });
    tbody.append(row);
  });
  table.append(thead, tbody);
  wrap.append(table);
  scoreSection.append(wrap);
  container.append(scoreSection);

  const details = document.createElement("section");
  details.className = "report-card";
  addText(details, "h2", "Training details");
  const columns = document.createElement("div");
  columns.className = "report-details";
  const preprocessing = document.createElement("div");
  addText(preprocessing, "h3", "Data preparation");
  const prepList = document.createElement("ul");
  report.preprocessing.forEach((step) => addText(prepList, "li", step));
  addText(preprocessing, "p", `Split: ${report.split.test_size * 100}% test, stratified, random seed ${report.split.random_state}.`, "report-note");
  preprocessing.append(prepList);
  const cvDetails = document.createElement("div");
  addText(cvDetails, "h3", "Extra Trees tuning");
  const cvList = document.createElement("ul");
  addText(cvList, "li", `${report.cross_validation.folds}-fold stratified cross-validation; score: ${report.cross_validation.search_scoring}.`);
  addText(cvList, "li", `Best parameters: ${JSON.stringify(report.best_extra_trees_parameters)}.`);
  addText(cvList, "li", `Saved model artifacts replaced: ${report.artifacts_saved ? "Yes" : "No (report-only run)"}.`);
  const tuning = document.createElement("div");
  cvDetails.append(cvList);
  tuning.append(cvDetails);
  columns.append(preprocessing, tuning);
  details.append(columns);

  const modelDetails = document.createElement("details");
  modelDetails.style.marginTop = "24px";
  addText(modelDetails, "summary", "Show classifier settings and input feature names");
  const modelHeading = document.createElement("h3");
  modelHeading.style.marginTop = "16px";
  modelHeading.textContent = "Compared classifiers";
  modelDetails.append(modelHeading);
  Object.entries(report.models_compared).forEach(([name, config]) => {
    const row = document.createElement("p");
    row.className = "report-note";
    row.textContent = `${name} (${config.estimator}): ${JSON.stringify(config.parameters)}`;
    modelDetails.append(row);
  });
  const featureHeading = document.createElement("h3");
  featureHeading.textContent = "Input features";
  modelDetails.append(featureHeading);
  addText(modelDetails, "p", report.feature_names.join(" · "), "report-note");
  const versionRow = document.createElement("p");
  versionRow.className = "report-note";
  versionRow.textContent = `Software versions: ${JSON.stringify(report.software_versions)}.`;
  modelDetails.append(versionRow);
  details.append(modelDetails);
  container.append(details);

  const caveat = document.createElement("section");
  caveat.className = "report-card";
  addText(caveat, "h2", "How to read these results");
  addText(caveat, "p", "This is a single exploratory holdout evaluation on the included dataset. It does not show how the model performs on new hospitals or populations, and it is not evidence that the model can diagnose PCOS. Do not use it to guide treatment or care.", "report-note");
  container.append(caveat);
}

fetch("model_results.json", { cache: "no-store" })
  .then((response) => {
    if (!response.ok) throw new Error("Evaluation report not found.");
    return response.json();
  })
  .then(renderReport)
  .catch(() => {
    container.replaceChildren();
    const error = document.createElement("div");
    error.className = "error-box";
    error.textContent = "No model evaluation report is available yet. From the project root, run: .\\.venv-clean\\Scripts\\python.exe train_model.py --report-only. Then reload this page.";
    container.append(error);
  });
