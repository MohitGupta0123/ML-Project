import streamlit as st
import pickle

# Load the ML model
model = joblib.load('travel_review_rating_model.pkl', 'rb')

# Streamlit app
def main():
    st.title("Travel Review Rating Predictor")

    # Ask user for ratings
    ratings = {}
    for category in ['Churches', 'Resorts', 'Beaches', 'Parks', 'Theatres', 'Museums', 'Malls', 'Zoo', 'Restaurants',
                     'Pubs_bars', 'Local_services', 'Burger_pizza_shops', 'Hotels_other_lodgings', 'Juice_bars',
                     'Art_galleries', 'Dance_clubs', 'Swimming_pools', 'Gyms', 'Bakeries', 'Beauty_spas', 'Cafes',
                     'View_points', 'Monuments', 'Gardens']:
        ratings[category] = st.slider(f"Rate {category} (1-5)", 1, 5)

    if st.button("Predict"):
        # Prepare input data
        input_data = [ratings[category] for category in ratings]

        # Predict the rating
        prediction = model.predict([input_data])[0]

        st.success(f"The predicted travel review rating is: {prediction}")

if __name__ == "__main__":
    main()
