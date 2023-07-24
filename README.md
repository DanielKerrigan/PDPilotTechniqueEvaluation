# PDPilot Techniques Evaluation

## Installation

```
conda env create -f environment.yml
conda activate pdpilot-eval
pip install -e .
```

## Schema

```typescript
{
  dataset: string;
  model: string;
  y_label: string;
  features: {
    name: string;
    kind: 'quantitative' | 'categorical';
    subkind: 'discrete' | 'continuous' | 'nominal' | 'ordinal';
    ordered: boolean;
    x_values: number[] | string[];
    ice_lines: number[][];
    pdp: number[];
    deviation: number;
    // Only for ordered features
    // Specifies the shape when the tolerance is 0.
    shape_tolerance_0: 'increasing' | 'decreasing';
    // Only for ordered features.
    // When the tolerance is greather than this threshold, the shape is mixed.
    tolerance_threshold: number;
    clustering: {
        method: string;
        num_clusters: number;
        labels: number[];
    }[];
  }[];
}
```
