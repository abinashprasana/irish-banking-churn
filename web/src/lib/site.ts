export const site = {
  name: "Atlantic Ledger",
  descriptor: "Irish banking churn and governed retention intelligence",
  title: "Atlantic Ledger — Irish Banking Churn & Governed Retention",
  description:
    "An explainable machine-learning case study connecting churn prediction, model evidence, counterfactual exploration, and a deterministic retention-policy gate.",
  url:
    process.env.NEXT_PUBLIC_SITE_URL ??
    "https://irish-banking-churn.vercel.app",
  labUrl:
    process.env.NEXT_PUBLIC_LAB_URL ??
    "https://abinashprasana-irish-banking-churn-app-aidovf.streamlit.app/",
  repositoryUrl: "https://github.com/abinashprasana/irish-banking-churn",
  author: "Abinash Prasana Selvanathan",
} as const;
