const sections = [
  {
    title: "01 · Basics & vitals",
    description: "General measurements and vital signs",
    fields: [
      ["age", "Age", "number", 25, 12, 60, 1, "years"],
      ["weight", "Weight", "number", 60, 30, 200, 0.1, "kg"],
      ["height", "Height", "number", 160, 100, 220, 0.1, "cm"],
      ["blood_group", "Blood group", "select", "B+", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]],
      ["pulse", "Pulse rate", "number", 72, 30, 220, 1, "bpm"],
      ["respiration", "Respiration rate", "number", 18, 5, 60, 1, "breaths/min"],
      ["hb", "Hemoglobin", "number", 12.5, 1, 25, 0.1, "g/dL"],
      ["bp_systolic", "Systolic blood pressure", "number", 120, 50, 250, 1, "mmHg"],
      ["bp_diastolic", "Diastolic blood pressure", "number", 80, 30, 150, 1, "mmHg"],
    ],
  },
  {
    title: "02 · Cycle & history",
    description: "Menstrual and reproductive history",
    fields: [
      ["cycle_regular", "Menstrual cycle", "select", "Regular", ["Regular", "Irregular"]],
      ["cycle_length", "Cycle length", "number", 28, 1, 120, 1, "days"],
      ["marriage_status", "Years married", "number", 0, 0, 80, 0.5, "years"],
      ["pregnant", "Currently pregnant", "select", "No", ["No", "Yes"]],
      ["abortions", "Number of abortions", "number", 0, 0, 30, 1, ""],
    ],
  },
  {
    title: "03 · Symptoms & lifestyle",
    description: "Select the closest answer for each item",
    fields: [
      ["weight_gain", "Recent weight gain", "select", "No", ["No", "Yes"]],
      ["hair_growth", "Excess hair growth", "select", "No", ["No", "Yes"]],
      ["skin_darkening", "Skin darkening", "select", "No", ["No", "Yes"]],
      ["hair_loss", "Hair loss", "select", "No", ["No", "Yes"]],
      ["pimples", "Acne / pimples", "select", "No", ["No", "Yes"]],
      ["fast_food", "Frequent fast food", "select", "No", ["No", "Yes"]],
      ["exercise", "Regular exercise", "select", "Yes", ["Yes", "No"]],
    ],
  },
  {
    title: "04 · Hormones & lab results",
    description: "Enter values from available laboratory reports",
    fields: [
      ["amh", "AMH", "number", 3.5, 0, 1000, 0.01, "ng/mL"],
      ["fsh", "FSH", "number", 4.5, 0, 10000, 0.01, "mIU/mL"],
      ["lh", "LH", "number", 4.5, 0, 10000, 0.01, "mIU/mL"],
      ["tsh", "TSH", "number", 2.5, 0, 10000, 0.01, "mIU/L"],
      ["prl", "Prolactin (PRL)", "number", 15, 0, 10000, 0.01, "ng/mL"],
      ["vitamin_d", "Vitamin D3", "number", 30, 0, 10000, 0.1, "ng/mL"],
      ["progesterone", "Progesterone", "number", 0.5, 0, 10000, 0.01, "ng/mL"],
      ["rbs", "Random blood sugar", "number", 95, 0, 10000, 0.1, "mg/dL"],
      ["beta_hcg_1", "β-HCG test I", "number", 1.99, 0, 1000000, 0.01, "mIU/mL"],
      ["beta_hcg_2", "β-HCG test II", "number", 1.99, 0, 1000000, 0.01, "mIU/mL"],
    ],
  },
  {
    title: "05 · Ultrasound & measurements",
    description: "Measurements in the units shown",
    fields: [
      ["waist", "Waist circumference", "number", 30, 1, 150, 0.1, "inches"],
      ["hip", "Hip circumference", "number", 36, 1, 180, 0.1, "inches"],
      ["follicle_left", "Follicle count · left", "number", 6, 0, 100, 1, ""],
      ["follicle_right", "Follicle count · right", "number", 6, 0, 100, 1, ""],
      ["avg_follicle_left", "Average follicle size · left", "number", 10, 0, 100, 0.1, "mm"],
      ["avg_follicle_right", "Average follicle size · right", "number", 10, 0, 100, 0.1, "mm"],
      ["endometrium", "Endometrium thickness", "number", 8, 0, 100, 0.1, "mm"],
    ],
  },
];

const formSections = document.querySelector("#form-sections");
const form = document.querySelector("#prediction-form");
const result = document.querySelector("#result");
const submitButton = document.querySelector("#submit-button");

function makeField(field) {
  const [name, label, type, value, optionsOrMin, max, step, unit] = field;
  const wrap = document.createElement("div");
  wrap.className = "field";
  const id = `field-${name}`;
  const title = document.createElement("label");
  title.htmlFor = id;
  title.textContent = label;
  wrap.append(title);

  let input;
  if (type === "select") {
    input = document.createElement("select");
    for (const optionValue of optionsOrMin) {
      const option = document.createElement("option");
      option.value = optionValue;
      option.textContent = optionValue;
      option.selected = optionValue === value;
      input.append(option);
    }
  } else {
    input = document.createElement("input");
    input.type = "number";
    input.min = String(optionsOrMin);
    input.max = String(max);
    input.step = String(step);
    input.value = String(value);
    input.required = true;
    input.inputMode = "decimal";
  }
  input.name = name;
  input.id = id;
  input.autocomplete = "off";
  wrap.append(input);
  if (unit) {
    const suffix = document.createElement("span");
    suffix.className = "unit";
    suffix.textContent = unit;
    wrap.append(suffix);
  }
  return wrap;
}

for (const section of sections) {
  const block = document.createElement("section");
  block.className = "input-section";
  const header = document.createElement("div");
  header.className = "section-heading";
  const title = document.createElement("h3");
  title.textContent = section.title;
  const description = document.createElement("p");
  description.textContent = section.description;
  header.append(title, description);
  const grid = document.createElement("div");
  grid.className = "field-grid";
  section.fields.forEach((field) => grid.append(makeField(field)));
  block.append(header, grid);
  formSections.append(block);
}

function renderResult(data) {
  const positive = data.prediction_class === 1;
  result.hidden = false;
  result.className = `result ${positive ? "result-positive" : "result-negative"}`;
  const heading = document.createElement("div");
  heading.className = "result-heading";
  const label = document.createElement("div");
  label.className = "eyebrow";
  label.textContent = "MODEL ESTIMATE";
  const title = document.createElement("h2");
  title.textContent = data.risk_level || (positive ? "Positive result" : "Negative result");
  const text = document.createElement("p");
  text.textContent = positive
    ? "The model indicates an elevated PCOS risk based on the provided values. This is not a diagnosis."
    : "The model indicates a lower PCOS risk based on the provided values. This does not rule out PCOS.";
  heading.append(label, title, text);

  const stats = document.createElement("div");
  stats.className = "result-stats";
  stats.append(stat("PCOS probability", `${data.pos_prob}%`), stat("Other class", `${data.neg_prob}%`));
  result.replaceChildren(heading, stats);

  if (Array.isArray(data.importances) && data.importances.length) {
    const subhead = document.createElement("h3");
    subhead.className = "importance-title";
    subhead.textContent = "Features with highest model importance";
    const list = document.createElement("ul");
    list.className = "importance-list";
    for (const item of data.importances) {
      const li = document.createElement("li");
      li.textContent = `${item.feature}: ${item.importance}%`;
      list.append(li);
    }
    result.append(subhead, list);
  }
}

function stat(label, value) {
  const item = document.createElement("div");
  item.className = "result-stat";
  const name = document.createElement("span");
  name.textContent = label;
  const amount = document.createElement("strong");
  amount.textContent = value;
  item.append(name, amount);
  return item;
}

form.addEventListener("reset", () => {
  result.hidden = true;
  result.replaceChildren();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  result.hidden = true;
  result.replaceChildren();
  if (!form.reportValidity()) return;

  const payload = Object.fromEntries(new FormData(form).entries());
  for (const [key, value] of Object.entries(payload)) {
    if (key !== "blood_group" && key !== "cycle_regular" && key !== "pregnant" &&
        key !== "weight_gain" && key !== "hair_growth" && key !== "skin_darkening" &&
        key !== "hair_loss" && key !== "pimples" && key !== "fast_food" && key !== "exercise") {
      payload[key] = Number(value);
    }
  }
  const baseUrl = document.querySelector("#api-url").value.trim().replace(/\/$/, "");
  submitButton.disabled = true;
  submitButton.querySelector("span:first-child").textContent = "Calculating…";

  try {
    const response = await fetch(`${baseUrl}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await response.json();
    if (!response.ok || !body.success) {
      throw new Error(body.detail || "The API could not complete the prediction.");
    }
    renderResult(body.data);
    result.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    result.hidden = false;
    result.className = "result result-error";
    const title = document.createElement("h2");
    title.textContent = "Unable to get an estimate";
    const message = document.createElement("p");
    message.textContent = error instanceof TypeError
      ? "Could not reach the prediction API. Start the FastAPI server and check the API URL above."
      : error.message;
    result.replaceChildren(title, message);
    result.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } finally {
    submitButton.disabled = false;
    submitButton.querySelector("span:first-child").textContent = "Calculate estimate";
  }
});
