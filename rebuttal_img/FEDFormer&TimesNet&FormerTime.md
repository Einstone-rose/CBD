| ACC. Result        | PS           | SRSCP1       | MI           | FM           | AWR          | SDA          | ECG5000      | FB           | UWare        |
| ------------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| SPANet(LP.)        | <u>18.22</u> | <u>86.72</u> | <u>64.50</u> | 56.40        | <u>98.52</u> | 97.98        | <u>94.51</u> | <u>79.57</u> | **95.99**    |
| SPANet(FT.)        | **23.27**    | 86.25        | **65.20**    | **60.60**    | **98.67**    | **99.44**    | **94.84**    | **80.83**    | <u>95.85</u> |
| FEDFormer(ICML'22) | 10.48        | 82.99        | 56.95        | 55.21        | 57.55        | 98.42        | 93.91        | 64.19        | 49.68        |
| TimesNet(ICLR'23)  | 14.62        | **87.60**    | 58.33        | 57.33        | 97.33        | 98.85        | 94.19        | 74.94        | 91.88        |
| FormerTime(WWW'23) | 15.40        | 86.69        | 61.33        | <u>60.00</u> | 97.00        | <u>98.94</u> | 94.21        | 79.01        | 90.84        |

| F1 Result          | PS           | SRSCP1       | MI           | FM           | AWR          | SAD          | ECG5000      | FB           | UWare        |
| ------------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| SPANet(LP.)        | <u>17.96</u> | <u>86.62</u> | <u>63.30</u> | 56.06        | <u>98.57</u> | 97.98        | 60.66        | <u>79.53</u> | **95.99**    |
| SPANet(FT.)        | **23.26**    | 86.18        | **64.65**    | **59.64**    | **98.73**    | **99.44**    | <u>60.73</u> | **80.78**    | <u>95.85</u> |
| FEDFormer(ICML'22) | 10.11        | 82.99        | 54.67        | 53.21        | 55.08        | 98.42        | 58.00        | 62.23        | 45.00        |
| TimesNet(ICLR'23)  | 14.40        | **87.16**    | 56.70        | <u>56.44</u> | 97.33        | 98.85        | 59.48        | 74.85        | 91.84        |
| FormerTime(WWW'23) | 15.08        | 86.38        | 63.07        | 54.51        | 96.96        | <u>98.94</u> | **60.78**    | 78.94        | 90.77        |

As the implementations of the respective methods in their repositories are all based on **supervised learning**, we will reproduce the results and compare them with our linear probing and fine-tuning results.

The implementations of FEDFormer and TimesNet are based on the open-source project [Time-Series-Library](https://github.com/thuml/Time-Series-Library).

The implementation of FormerTime is based on [FormerTime](https://github.com/Mingyue-Cheng/FormerTime).

Part of training logs can be found [here](https://anonymous.4open.science/r/SPANet-0D21/baseline_run_log/).