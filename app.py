import streamlit as st
import pickle
import numpy as np

np.random.seed(3)
np.set_printoptions(precision=3)
from bokeh.plotting import figure

# # Use the full page instead of a narrow central column
st.set_page_config(layout="wide")

# Space out the maps so the first one is 2x the size of the other three
c1, c2 = st.beta_columns((2, 1))

st.title("Discharge Barrier Quantification")
st.title("Anomaly Detector for Excess Days (POC)")


def main():
    """Streamlit driver code"""
    # %% Load model
    classifier = pickle.load(open("./models/DecisionTreeClassifierModel.pkl", "rb"))

    # %% Form
    with st.form(key="my_form2"):

        # %% Get feature from user
        st.sidebar.markdown("**Input the patient details below**")

        age = int(st.sidebar.number_input("age", value=51, step=1))
        SLD = int(
            st.sidebar.slider(
                label="SLD",
                min_value=0,
                max_value=1,
                value=0,
                step=1,
            )
        )
        Restraints = int(
            st.sidebar.slider(
                label="Restraints", min_value=0, max_value=1, value=0, step=1
            )
        )
        
        Not_Medically_Ready = int(
            st.sidebar.slider(
                label="Not Medically Ready",
                min_value=0,
                max_value=1,
                value=0,
                step=1,
            )
        )
        Authorizations = int(
            st.sidebar.slider(
                label="Authorizations", min_value=0, max_value=1, value=0, step=1
            )
        )
        Delayed_Pending_Procedure = int(
            st.sidebar.slider(
                label="Delayed/Pending Procedure", min_value=0, max_value=1, value=0, step=1
            )
        )
        Transfer_Placement = int(
            st.sidebar.slider(
                label="Transfer/Placement", min_value=0, max_value=1, value=0, step=1
            )
        )
        Diet = int(
            st.sidebar.slider(
                label="Diet", min_value=0, max_value=1, value=0, step=1
            )
        )       
        Pending_Dispostion = int(
            st.sidebar.slider(
                label="Pending Dispostion", min_value=0, max_value=1, value=0, step=1
            )
        ) 
        Consults_Pending = int(
            st.sidebar.slider(
                label="Consults Pending", min_value=0, max_value=1, value=1, step=1
            )
        ) 
        Oxygen = int(
            st.sidebar.slider(
                label="Oxygen", min_value=0, max_value=1, value=0, step=1
            )
        ) 
        Diagnostic_Test = int(
            st.sidebar.slider(
                label="Diagnostic Test", min_value=0, max_value=1, value=0, step=1
            )
        ) 
        gender = int(
            st.sidebar.slider(
                label="Gender (0 = Female; 1= Male)", min_value=0, max_value=1, value=0, step=1
            )
        ) 
        Awaiting_Patient_Decision_Maker = int(
            st.sidebar.slider(
                label="Awaiting Patient/Decision Maker", min_value=0, max_value=1, value=0, step=1
            )
        ) 
        Bowel_Movement = int(
            st.sidebar.slider(
                label="Bowel Movement", min_value=0, max_value=1, value=0, step=1
            )
        ) 
        COVID_19 = int(
            st.sidebar.slider(
                label="COVID 19", min_value=0, max_value=1, value=0, step=1
            )
        ) 
        Discharge = int(
            st.sidebar.slider(
                label="Discharge", min_value=0, max_value=1, value=0, step=1
            )
        ) 
        Trauma = int(
            st.sidebar.slider(
                label="Trauma", min_value=0, max_value=1, value=0, step=1
            )
        )
        GMLOS = int(st.sidebar.number_input("GMLOS", value=4, step=1))
        
        st.markdown("Input the patient barriers on the left sidebar and click submit button below to analyze")

        feature = np.array(
            [
                age,
                SLD,
                Restraints,
                Not_Medically_Ready,
                GMLOS,
                Pending_Dispostion,
                Transfer_Placement,
                Authorizations,
                gender,
                Oxygen,
                Discharge,
                Delayed_Pending_Procedure,
                Diet,
                Diagnostic_Test,
                Consults_Pending,
                Awaiting_Patient_Decision_Maker,
                Bowel_Movement,
                COVID_19,
                Trauma
            ]
        ).reshape(1, -1)

        submit_button = st.form_submit_button(label="Submit")

        if submit_button:
            # %% prediction
            prediction = classifier.predict(feature)[0]

            if prediction == 1:
                st.error("The patient is likely to have HIGH 🔴 excess days.")

            else:
                st.success("No excess days 🟢 detected.")

    ckbox = st.checkbox(label="View details", value=False)



    if ckbox:
        st.json(dict(zip([0, 1], list(classifier.predict_proba(feature)[0]))))

        # %% Feature importance graph
        feature_names = np.array(
            [
                'age',
                'SLD',
                'Restraints',
                'Not Medically Ready',
                'GMLOS',
                'Pending Dispostion',
                'Transfer/Placement',
                'Authorizations',
                'gender',
                'Oxygen',
                'Discharge',
                'Delayed/Pending Procedure',
                'Diet',
                'Diagnostic Test',
                'Consults Pending',
                'Awaiting Patient/Decision Maker',
                'Bowel Movement',
                'COVID 19',
                'Trauma'
            ]
        )

        importance = np.array(
            [int(round(x * 100)) for x in classifier.feature_importances_]
        )
        importance = np.concatenate(
            (
                importance.reshape(len(importance), 1),
                feature_names.reshape(len(feature_names), 1),
            ),
            1,
        )
        importance = importance[importance[:, 0].argsort()[::-1]]

        # plot feature importance for CART
        x_range = list(importance[:, 1])
        y_range = list(map(lambda x: float(x), importance[:, 0]))

        x_range_sorted = sorted(x_range, key=lambda x: y_range[x_range.index(x)])

        p = figure(
            title="Relative Feature Importance Bar Chart ",
            x_range=x_range_sorted,
            x_axis_label="Features",
            y_axis_label="Importance (%)",
            match_aspect=True,
            aspect_ratio=10 / 4,
        )

        p.vbar(x=x_range, top=y_range, color="#008000", width=0.9)
        p.y_range.start = 0
        p.xaxis.major_label_orientation = 1
        st.bokeh_chart(p, use_container_width=True)


if __name__ == "__main__":

    main()

# # HA