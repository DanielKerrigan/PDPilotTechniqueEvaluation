# Labeling tool

This directory contains the web application we made for labeling the shape of PDPs. It also contains a tool for comparing ICE plot clustering results, but we ultimately decided not to use that approach for the paper.

You can access the tool at [https://pdpilot-technique-evaluation.vercel.app](https://pdpilot-technique-evaluation.vercel.app).

To use it, click the "Filtering" button and then upload [a JSON file containing PDPs](../filtering/small-pdps.json).

To run it locally, you can use the following commands:

```bash
npm install
npm run build
npm run preview
```
