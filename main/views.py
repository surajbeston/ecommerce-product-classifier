from django.shortcuts import render
from joblib import load

from django.conf import settings

text_clf = load(settings.BASE_DIR.parent  / "ecommerce_classifier/classification_model1.joblib")
encoder = load(settings.BASE_DIR.parent  / "ecommerce_classifier/category_encoder1.joblib")


def home(request):
    name = request.GET.get("name", None)
    description = request.GET.get("description", None)

    if name and description:
        classify_text = name + description
    elif name:
        classify_text = name
    elif description:
        classify_text = description
    else:
        return render(request , "home.html", {})

    categories = encoder.inverse_transform([text_clf.predict([classify_text])])

    return render(request , "home.html",  {"categories": categories[0]})