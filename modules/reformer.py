def Reformer(classifier, team_obj, filtered_images, labels):
    # evaluates the classifier on the filtered images
    original = classifier.evaluate(filtered_images, labels)[1]

    # chooses one autoencoder randomly as a reformer
    autoencoder = team_obj.load_autoencoder(team_obj.get_team(number=1)[0])
    
    # reforms the filtered images
    reformed_images = autoencoder.predict(filtered_images)

    # evaluates the reformed images on the classifier
    reformed = classifier.evaluate(reformed_images, labels)[1]
    return original, reformed
    
        