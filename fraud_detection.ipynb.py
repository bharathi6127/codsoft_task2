{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPYm3MI2rurpx+4smUK2Kpm",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/bharathi6127/codsoft_task2/blob/main/fraud_detection.ipynb.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#CREDIT CARD FRAUD DETECTION\n",
        "\n",
        "Build a model to detect fraudulent credit card transactions. Use a\n",
        "dataset containing information about credit card transactions, and\n",
        "experiment with algorithms like Logistic Regression, Decision Trees,\n",
        "or Random Forests to classify transactions as fraudulent or legitimate."
      ],
      "metadata": {
        "id": "hb27tguy2dXj"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "#Import Libraries"
      ],
      "metadata": {
        "id": "2ehAG-iXADiE"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score\n",
        "from sklearn.model_selection import GridSearchCV\n"
      ],
      "metadata": {
        "id": "qEccwlQK91Hb"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Load datasets"
      ],
      "metadata": {
        "id": "hq20pjuEkdUU"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "train_data = pd.read_csv('/content/train_fraud.csv')\n",
        "test_data = pd.read_csv('/content/test_fraud.csv')"
      ],
      "metadata": {
        "id": "HSwrNBzJ91Tx"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Verify column names"
      ],
      "metadata": {
        "id": "5mORN5ldASiy"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Train Data Columns: \", train_data.columns)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "kSTtUBDtA0Rx",
        "outputId": "0e494708-f382-4a3c-a234-a3ae71b06570"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Train Data Columns:  Index(['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'category',\n",
            "       'amt', 'first', 'last', 'gender', 'street', 'city', 'state', 'zip',\n",
            "       'lat', 'long', 'city_pop', 'job', 'dob', 'trans_num', 'unix_time',\n",
            "       'merch_lat', 'merch_long', 'is_fraud'],\n",
            "      dtype='object')\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Test Data Columns: \", test_data.columns)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bkdEq36XA6P3",
        "outputId": "e8913cae-bfc3-4bb6-a74c-c23427904b5b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Test Data Columns:  Index(['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'category',\n",
            "       'amt', 'first', 'last', 'gender', 'street', 'city', 'state', 'zip',\n",
            "       'lat', 'long', 'city_pop', 'job', 'dob', 'trans_num', 'unix_time',\n",
            "       'merch_lat', 'merch_long', 'is_fraud'],\n",
            "      dtype='object')\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(train_data.info())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NzhYPaLD-1XK",
        "outputId": "46eda9d4-b7a0-4929-ca36-761982f7be61"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 57042 entries, 0 to 57041\n",
            "Data columns (total 23 columns):\n",
            " #   Column                 Non-Null Count  Dtype  \n",
            "---  ------                 --------------  -----  \n",
            " 0   Unnamed: 0             57042 non-null  int64  \n",
            " 1   trans_date_trans_time  57042 non-null  object \n",
            " 2   cc_num                 57042 non-null  float64\n",
            " 3   merchant               57042 non-null  object \n",
            " 4   category               57042 non-null  object \n",
            " 5   amt                    57042 non-null  float64\n",
            " 6   first                  57042 non-null  object \n",
            " 7   last                   57042 non-null  object \n",
            " 8   gender                 57042 non-null  object \n",
            " 9   street                 57042 non-null  object \n",
            " 10  city                   57042 non-null  object \n",
            " 11  state                  57042 non-null  object \n",
            " 12  zip                    57042 non-null  int64  \n",
            " 13  lat                    57042 non-null  float64\n",
            " 14  long                   57042 non-null  float64\n",
            " 15  city_pop               57042 non-null  int64  \n",
            " 16  job                    57042 non-null  object \n",
            " 17  dob                    57042 non-null  object \n",
            " 18  trans_num              57042 non-null  object \n",
            " 19  unix_time              57041 non-null  float64\n",
            " 20  merch_lat              57041 non-null  float64\n",
            " 21  merch_long             57041 non-null  float64\n",
            " 22  is_fraud               57041 non-null  float64\n",
            "dtypes: float64(8), int64(3), object(12)\n",
            "memory usage: 10.0+ MB\n",
            "None\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(test_data.info())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6lpTYZx8Ag5Y",
        "outputId": "b35e02eb-75cb-4966-ede8-49cff53047f4"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 16316 entries, 0 to 16315\n",
            "Data columns (total 23 columns):\n",
            " #   Column                 Non-Null Count  Dtype  \n",
            "---  ------                 --------------  -----  \n",
            " 0   Unnamed: 0             16316 non-null  int64  \n",
            " 1   trans_date_trans_time  16316 non-null  object \n",
            " 2   cc_num                 16316 non-null  float64\n",
            " 3   merchant               16316 non-null  object \n",
            " 4   category               16316 non-null  object \n",
            " 5   amt                    16316 non-null  float64\n",
            " 6   first                  16316 non-null  object \n",
            " 7   last                   16315 non-null  object \n",
            " 8   gender                 16315 non-null  object \n",
            " 9   street                 16315 non-null  object \n",
            " 10  city                   16315 non-null  object \n",
            " 11  state                  16315 non-null  object \n",
            " 12  zip                    16315 non-null  float64\n",
            " 13  lat                    16315 non-null  float64\n",
            " 14  long                   16315 non-null  float64\n",
            " 15  city_pop               16315 non-null  float64\n",
            " 16  job                    16315 non-null  object \n",
            " 17  dob                    16315 non-null  object \n",
            " 18  trans_num              16315 non-null  object \n",
            " 19  unix_time              16315 non-null  float64\n",
            " 20  merch_lat              16315 non-null  float64\n",
            " 21  merch_long             16315 non-null  float64\n",
            " 22  is_fraud               16315 non-null  float64\n",
            "dtypes: float64(10), int64(1), object(12)\n",
            "memory usage: 2.9+ MB\n",
            "None\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(train_data.head())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2zcxcDf1-1ag",
        "outputId": "01414dbd-0cf5-4035-9661-d357d36bc056"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "   Unnamed: 0 trans_date_trans_time        cc_num  \\\n",
            "0           0      01-01-2019 00:00  2.703190e+15   \n",
            "1           1      01-01-2019 00:00  6.304230e+11   \n",
            "2           2      01-01-2019 00:00  3.885950e+13   \n",
            "3           3      01-01-2019 00:01  3.534090e+15   \n",
            "4           4      01-01-2019 00:03  3.755340e+14   \n",
            "\n",
            "                             merchant       category     amt      first  \\\n",
            "0          fraud_Rippin, Kub and Mann       misc_net    4.97   Jennifer   \n",
            "1     fraud_Heller, Gutmann and Zieme    grocery_pos  107.23  Stephanie   \n",
            "2                fraud_Lind-Buckridge  entertainment  220.11     Edward   \n",
            "3  fraud_Kutch, Hermiston and Farrell  gas_transport   45.00     Jeremy   \n",
            "4                 fraud_Keeling-Crist       misc_pos   41.96      Tyler   \n",
            "\n",
            "      last gender                        street  ...      lat      long  \\\n",
            "0    Banks      F                561 Perry Cove  ...  36.0788  -81.1781   \n",
            "1     Gill      F  43039 Riley Greens Suite 393  ...  48.8878 -118.2105   \n",
            "2  Sanchez      M      594 White Dale Suite 530  ...  42.1808 -112.2620   \n",
            "3    White      M   9443 Cynthia Court Apt. 038  ...  46.2306 -112.1138   \n",
            "4   Garcia      M              408 Bradley Rest  ...  38.4207  -79.4629   \n",
            "\n",
            "   city_pop                                job         dob  \\\n",
            "0      3495          Psychologist, counselling  09-03-1988   \n",
            "1       149  Special educational needs teacher  21-06-1978   \n",
            "2      4154        Nature conservation officer  19-01-1962   \n",
            "3      1939                    Patent attorney  12-01-1967   \n",
            "4        99     Dance movement psychotherapist  28-03-1986   \n",
            "\n",
            "                          trans_num     unix_time  merch_lat  merch_long  \\\n",
            "0  0b242abb623afc578575680df30655b9  1.325376e+09  36.011293  -82.048315   \n",
            "1  1f76529f8574734946361c461b024d99  1.325376e+09  49.159047 -118.186462   \n",
            "2  a1a22d70485983eac12b5b88dad1cf95  1.325376e+09  43.150704 -112.154481   \n",
            "3  6b849c168bdad6f867558c3793159a81  1.325376e+09  47.034331 -112.561071   \n",
            "4  a41d7549acf90789359a9aa5346dcb46  1.325376e+09  38.674999  -78.632459   \n",
            "\n",
            "   is_fraud  \n",
            "0       0.0  \n",
            "1       0.0  \n",
            "2       0.0  \n",
            "3       0.0  \n",
            "4       0.0  \n",
            "\n",
            "[5 rows x 23 columns]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(test_data.head())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "dQuM780BBP8H",
        "outputId": "af684e1f-9185-4c8d-fd2b-7ee3f545d114"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "   Unnamed: 0 trans_date_trans_time        cc_num  \\\n",
            "0           0      21-06-2020 12:14  2.291160e+15   \n",
            "1           1      21-06-2020 12:14  3.573030e+15   \n",
            "2           2      21-06-2020 12:14  3.598220e+15   \n",
            "3           3      21-06-2020 12:15  3.591920e+15   \n",
            "4           4      21-06-2020 12:15  3.526830e+15   \n",
            "\n",
            "                               merchant        category    amt   first  \\\n",
            "0                 fraud_Kirlin and Sons   personal_care   2.86    Jeff   \n",
            "1                  fraud_Sporer-Keebler   personal_care  29.84  Joanne   \n",
            "2  fraud_Swaniawski, Nitzsche and Welch  health_fitness  41.28  Ashley   \n",
            "3                     fraud_Haley Group        misc_pos  60.05   Brian   \n",
            "4                 fraud_Johnston-Casper          travel   3.19  Nathan   \n",
            "\n",
            "       last gender                       street  ...      lat      long  \\\n",
            "0   Elliott      M            351 Darlene Green  ...  33.9659  -80.9355   \n",
            "1  Williams      F             3638 Marsh Union  ...  40.3207 -110.4360   \n",
            "2     Lopez      F         9333 Valentine Point  ...  40.6729  -73.5365   \n",
            "3  Williams      M  32941 Krystal Mill Apt. 552  ...  28.5697  -80.8191   \n",
            "4    Massey      M     5783 Evan Roads Apt. 465  ...  44.2529  -85.0170   \n",
            "\n",
            "   city_pop                     job         dob  \\\n",
            "0  333497.0     Mechanical engineer  19-03-1968   \n",
            "1     302.0  Sales professional, IT  17-01-1990   \n",
            "2   34496.0       Librarian, public  21-10-1970   \n",
            "3   54767.0            Set designer  25-07-1987   \n",
            "4    1126.0      Furniture designer  06-07-1955   \n",
            "\n",
            "                          trans_num     unix_time  merch_lat  merch_long  \\\n",
            "0  2da90c7d74bd46a0caf3777415b3ebd3  1.371817e+09  33.986391  -81.200714   \n",
            "1  324cc204407e99f51b0d6ca0055005e7  1.371817e+09  39.450498 -109.960431   \n",
            "2  c81755dbbbea9d5c77f094348a7579be  1.371817e+09  40.495810  -74.196111   \n",
            "3  2159175b9efe66dc301f149d3d5abf8c  1.371817e+09  28.812398  -80.883061   \n",
            "4  57ff021bd3f328f8738bb535c302a31b  1.371817e+09  44.959148  -85.884734   \n",
            "\n",
            "   is_fraud  \n",
            "0       0.0  \n",
            "1       0.0  \n",
            "2       0.0  \n",
            "3       0.0  \n",
            "4       0.0  \n",
            "\n",
            "[5 rows x 23 columns]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(train_data.tail())\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "pYWfT29C_GF8",
        "outputId": "ba54e7aa-b6a5-488c-f485-a47a34bce1a0"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "       Unnamed: 0 trans_date_trans_time        cc_num  \\\n",
            "57037       57037      03-02-2019 19:30  6.539650e+15   \n",
            "57038       57038      03-02-2019 19:32  2.131580e+14   \n",
            "57039       57039      03-02-2019 19:32  6.012000e+15   \n",
            "57040       57040      03-02-2019 19:32  3.801440e+13   \n",
            "57041       57041      03-02-2019 19:32  4.642890e+12   \n",
            "\n",
            "                           merchant        category     amt   first      last  \\\n",
            "57037  fraud_Torp, Muller and Borer  health_fitness   36.73  Samuel     Short   \n",
            "57038        fraud_Eichmann-Kilback            home   13.77    Tara  Campbell   \n",
            "57039      fraud_Altenwerth-Kilback            home  179.06  Ronald    Carson   \n",
            "57040              fraud_Renner Ltd            home   12.82   Sonya    Jensen   \n",
            "57041              fraud_Schumm PLC    shopping_net    9.52   Eddie    Mendez   \n",
            "\n",
            "      gender                      street  ...      lat      long  city_pop  \\\n",
            "57037      M  23383 Denise Pine Apt. 099  ...  38.6865 -121.3494    757530   \n",
            "57038      F  05050 Rogers Well Apt. 439  ...  41.6060 -109.2300     27971   \n",
            "57039      M             870 Rocha Drive  ...  40.9918  -73.9800      4664   \n",
            "57040      F         4470 Jillian Courts  ...  43.8967  -89.8219      3508   \n",
            "57041      M   1831 Faith View Suite 653  ...  40.7491  -95.0380      7297   \n",
            "\n",
            "                                   job         dob  \\\n",
            "57037                   Soil scientist  19-09-1966   \n",
            "57038                  Music therapist  01-08-1984   \n",
            "57039         Radiographer, diagnostic  30-06-1965   \n",
            "57040  Sport and exercise psychologist  11-08-1946   \n",
            "57041                       IT trainer  13-07-1990   \n",
            "\n",
            "                              trans_num     unix_time  merch_lat  merch_long  \\\n",
            "57037  cc80340ed0235e4208dcd144e13339f7  1.328297e+09  39.221351 -121.311926   \n",
            "57038  c2125b2e6a7f5cbecc8b6124e723f8f6  1.328298e+09  41.253971 -110.067070   \n",
            "57039  22ae5b8b2b6b15ee1a610bcf0c427be0  1.328298e+09  41.922753  -73.698126   \n",
            "57040  20ad09a8009adccad1cee8331277f9c3  1.328298e+09  44.407264  -89.038198   \n",
            "57041     37396b343de26642ca92492b27ea2           NaN        NaN         NaN   \n",
            "\n",
            "       is_fraud  \n",
            "57037       0.0  \n",
            "57038       0.0  \n",
            "57039       0.0  \n",
            "57040       0.0  \n",
            "57041       NaN  \n",
            "\n",
            "[5 rows x 23 columns]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(test_data.tail())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ovseFjBgBZSL",
        "outputId": "0252fcff-c3ee-4411-c9bd-e399c76426ca"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "       Unnamed: 0 trans_date_trans_time        cc_num                merchant  \\\n",
            "16311       16311      27-06-2020 00:31  4.482430e+15  fraud_Schmidt and Sons   \n",
            "16312       16312      27-06-2020 00:31  3.703490e+14      fraud_Hudson-Grady   \n",
            "16313       16313      27-06-2020 00:34  3.458320e+14    fraud_Murray-Smitham   \n",
            "16314       16314      27-06-2020 00:35  3.560320e+15      fraud_Hickle Group   \n",
            "16315       16315      27-06-2020 00:35  6.011150e+15        fraud_Herzog Ltd   \n",
            "\n",
            "           category    amt        first     last gender  \\\n",
            "16311  shopping_net   6.66       Leslie     Ford      F   \n",
            "16312  shopping_pos   7.24  Christopher    Gomez      M   \n",
            "16313   grocery_pos  68.37        Jason  Mcmahon      M   \n",
            "16314  shopping_pos   9.43      William  Skinner      M   \n",
            "16315      misc_pos   2.08         Terr      NaN    NaN   \n",
            "\n",
            "                             street  ...      lat     long  city_pop  \\\n",
            "16311          4938 Hatfield Course  ...  38.8265 -82.1364     642.0   \n",
            "16312              950 Dunn Squares  ...  30.5668 -90.4820    8512.0   \n",
            "16313  6385 Donald Square Suite 429  ...  38.8029 -77.2116  104396.0   \n",
            "16314        524 Wu Spurs Suite 894  ...  34.4793 -87.4769    1312.0   \n",
            "16315                           NaN  ...      NaN      NaN       NaN   \n",
            "\n",
            "                                        job         dob  \\\n",
            "16311            Building services engineer  30-08-1946   \n",
            "16312  Accountant, chartered public finance  03-09-1951   \n",
            "16313                   Production engineer  20-11-1950   \n",
            "16314                   Librarian, academic  01-02-1955   \n",
            "16315                                   NaN         NaN   \n",
            "\n",
            "                              trans_num     unix_time  merch_lat merch_long  \\\n",
            "16311  2166fed0d516d9304e0153776edce1a5  1.372293e+09  38.827234 -81.940621   \n",
            "16312  aadd946642da1cfe0e941dbab845c01b  1.372293e+09  31.136023 -90.619269   \n",
            "16313  f281b35381a193e310da072d62290350  1.372293e+09  39.750399 -76.829325   \n",
            "16314  8f64990b4c57b3a80bba6b0f4f220bb6  1.372293e+09  35.318853 -88.167661   \n",
            "16315                               NaN           NaN        NaN        NaN   \n",
            "\n",
            "       is_fraud  \n",
            "16311       0.0  \n",
            "16312       0.0  \n",
            "16313       0.0  \n",
            "16314       0.0  \n",
            "16315       NaN  \n",
            "\n",
            "[5 rows x 23 columns]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(train_data.shape)\n",
        "print(test_data.shape)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "J6kGhdJH_NwI",
        "outputId": "16589843-aff6-423c-c785-1c093875606e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(57042, 23)\n",
            "(16316, 23)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Drop non-numeric columns that can't be used directly for modeling\n",
        "columns_to_drop = ['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'category',\n",
        "                   'first', 'last', 'gender', 'street', 'city', 'state', 'zip', 'lat', 'long',\n",
        "                   'city_pop', 'job', 'dob', 'trans_num', 'unix_time', 'merch_lat', 'merch_long']\n",
        "\n",
        "X_train = train_data.drop(columns_to_drop + ['is_fraud'], axis=1)\n",
        "y_train = train_data['is_fraud']\n",
        "\n",
        "X_test = test_data.drop(columns_to_drop + ['is_fraud'], axis=1)\n",
        "y_test = test_data['is_fraud']\n"
      ],
      "metadata": {
        "id": "PRYwhbHy-ApX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Handle missing values in features\n",
        "X_train = X_train.fillna(X_train.mean())\n",
        "X_test = X_test.fillna(X_test.mean())\n",
        "\n",
        "# Handle missing values in target variable\n",
        "y_train = y_train.fillna(y_train.mode()[0])\n",
        "y_test = y_test.fillna(y_test.mode()[0])\n",
        "\n",
        "# Scale the 'amt' feature\n",
        "scaler = StandardScaler()\n",
        "X_train['amt'] = scaler.fit_transform(X_train[['amt']])\n",
        "X_test['amt'] = scaler.transform(X_test[['amt']])\n"
      ],
      "metadata": {
        "id": "ufGjUE8F-A6A"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Logistic Regression"
      ],
      "metadata": {
        "id": "00h7-fjUktre"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "log_reg = LogisticRegression(max_iter=1000)\n",
        "log_reg.fit(X_train, y_train)\n",
        "y_pred_log_reg = log_reg.predict(X_test)\n",
        "print(\"Logistic Regression:\")\n",
        "print(confusion_matrix(y_test, y_pred_log_reg))\n",
        "print(classification_report(y_test, y_pred_log_reg, target_names=['fraudulent', 'legitimate']))\n",
        "print('ROC AUC:', roc_auc_score(y_test, y_pred_log_reg))\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "JC5T8hbI-A_g",
        "outputId": "c590f18d-a0c3-43f9-8a35-77c7209f5216"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Logistic Regression:\n",
            "[[16231    32]\n",
            " [   53     0]]\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "  fraudulent       1.00      1.00      1.00     16263\n",
            "  legitimate       0.00      0.00      0.00        53\n",
            "\n",
            "    accuracy                           0.99     16316\n",
            "   macro avg       0.50      0.50      0.50     16316\n",
            "weighted avg       0.99      0.99      0.99     16316\n",
            "\n",
            "ROC AUC: 0.4990161716780422\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Decision Tree"
      ],
      "metadata": {
        "id": "urev8dO9kzfg"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "tree = DecisionTreeClassifier()\n",
        "tree.fit(X_train, y_train)\n",
        "y_pred_tree = tree.predict(X_test)\n",
        "print(\"Decision Tree:\")\n",
        "print(confusion_matrix(y_test, y_pred_tree))\n",
        "print(classification_report(y_test, y_pred_tree, target_names=['fraudulent', 'legitimate']))\n",
        "print('ROC AUC:', roc_auc_score(y_test, y_pred_tree))\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "pDX2GLQ--BBs",
        "outputId": "749159fe-3ad0-4dcd-91fe-ad94df15839c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Decision Tree:\n",
            "[[16191    72]\n",
            " [   29    24]]\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "  fraudulent       1.00      1.00      1.00     16263\n",
            "  legitimate       0.25      0.45      0.32        53\n",
            "\n",
            "    accuracy                           0.99     16316\n",
            "   macro avg       0.62      0.72      0.66     16316\n",
            "weighted avg       1.00      0.99      0.99     16316\n",
            "\n",
            "ROC AUC: 0.7242014806152175\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Random Forest"
      ],
      "metadata": {
        "id": "aGAds9Uak47u"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "forest = RandomForestClassifier()\n",
        "forest.fit(X_train, y_train)\n",
        "y_pred_forest = forest.predict(X_test)\n",
        "print(\"Random Forest:\")\n",
        "print(confusion_matrix(y_test, y_pred_forest))\n",
        "print(classification_report(y_test, y_pred_forest, target_names=['fraudulent', 'legitimate']))\n",
        "print('ROC AUC:', roc_auc_score(y_test, y_pred_forest))\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sUahaYlC-BGM",
        "outputId": "f6404614-fb0e-4266-e138-279376ff7a3d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Random Forest:\n",
            "[[16190    73]\n",
            " [   28    25]]\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "  fraudulent       1.00      1.00      1.00     16263\n",
            "  legitimate       0.26      0.47      0.33        53\n",
            "\n",
            "    accuracy                           0.99     16316\n",
            "   macro avg       0.63      0.73      0.66     16316\n",
            "weighted avg       1.00      0.99      0.99     16316\n",
            "\n",
            "ROC AUC: 0.7336046982443073\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Hyperparameter Tuning for Random Forest"
      ],
      "metadata": {
        "id": "rfmsPdA9lArs"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "param_grid = {\n",
        "    'n_estimators': [50, 100, 200],\n",
        "    'max_features': ['auto', 'sqrt', 'log2'],\n",
        "    'max_depth': [None, 10, 20, 30]\n",
        "}\n",
        "grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5, scoring='roc_auc')\n",
        "grid_search.fit(X_train, y_train)\n",
        "\n",
        "best_model = grid_search.best_estimator_\n",
        "y_pred_best = best_model.predict(X_test)\n",
        "\n",
        "print(\"Tuned Random Forest:\")\n",
        "print(confusion_matrix(y_test, y_pred_best))\n",
        "print(classification_report(y_test, y_pred_best))\n",
        "print('Best ROC AUC:', roc_auc_score(y_test, y_pred_best))\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qI3ymZfn-BJm",
        "outputId": "ae63ef96-9840-4b61-b68b-c2cb9e68ef3e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_forest.py:424: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.\n",
            "  warn(\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Tuned Random Forest:\n",
            "[[16216    47]\n",
            " [   32    21]]\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "         0.0       1.00      1.00      1.00     16263\n",
            "         1.0       0.31      0.40      0.35        53\n",
            "\n",
            "    accuracy                           1.00     16316\n",
            "   macro avg       0.65      0.70      0.67     16316\n",
            "weighted avg       1.00      1.00      1.00     16316\n",
            "\n",
            "Best ROC AUC: 0.6966682096992943\n"
          ]
        }
      ]
    }
  ]
}