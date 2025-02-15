async function predictSentiment(event) {
    event.preventDefault(); // Empêche le rechargement de la page lors de la soumission du formulaire

    const tweetText = document.querySelector('textarea[name="tweet_to_predict"]').value;
    const resultDiv = document.getElementById('result');

    // Réinitialiser le résultat avant de soumettre
    resultDiv.textContent = ''; 

    // Envoi des données à l'API Flask
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ tweet_to_predict: tweetText })
    });

    const data = await response.json();

    // Affichage du résultat
    resultDiv.textContent = `Résultat : ${data.prediction}`;
    
    // Affichage des boutons de feedback
    document.querySelector('.feedback-buttons').style.display = 'flex';

    // Sauvegarde le tweet pour le feedback ultérieur
    document.querySelector('.feedback-buttons').dataset.tweet = tweetText;
}

async function sendFeedback(isPositive) {
    const tweetText = document.querySelector('.feedback-buttons').dataset.tweet;
    const feedbackRoute = isPositive ? '/feedbackpositif' : '/feedbacknegatif';

    // Envoi des feedbacks à l'API Flask
    await fetch(feedbackRoute, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ tweet_to_predict: tweetText })
    });

    // Afficher la notification de remerciement
    showNotification();
}

function showNotification() {
    const notificationDiv = document.getElementById('notification');
    notificationDiv.style.display = 'block';

    document.getElementById("tweet").value = "";

    const resultDiv = document.getElementById('result');
    resultDiv.textContent = ''; 
  
    document.querySelector('.feedback-buttons').style.display = 'none';

    // Fermer automatiquement la notification après 5 secondes
    setTimeout(() => {
        notificationDiv.style.display = 'none';
    }, 2000);
}

function closeNotification() {
    document.getElementById('notification').style.display = 'none';
}