import utils.helpers as helpers
import numpy as np

def poll_votes_each_x(x, y, x_marks, thresholds, reduction_models):
    label = "BENIGN" if y == 1 else "ADVERSARIAL"
    print("\rInput image 'x' {0} sample.".format(label))
    # checks whether 'x' is adversarial or not by votating using each reduction model's threshold.
    v_adv = 0
    v_leg = 0

    for j in range(reduction_models):
        #print("'x' input RE: {0}\nThreshold: {1}".format(x_marks[j][i], thresholds[j]))
        if x_marks[j] < thresholds[j]:
            v_leg = v_leg + 1
        else: v_adv = v_adv + 1
    
    ans = 1 if v_leg > v_adv else 0
    # helpers.assign_confusion_matrix(cm, y, ans)
    return ans

def poll_votes(x, y, x_marks, thresholds, reduction_models):
    y_pred = np.zeros((len(y)))
    filtered_images = []

    for i in range(len(x)):
            # checks whether 'x' is adversarial or not by voting using each reduction model's threshold.
            v_adv = 0
            v_leg = 0

            for j in range(reduction_models):
                # print("'x' input RE: {0}\nThreshold: {1}".format(x_marks[j][i], thresholds[j]))
                if x_marks[j][i] < thresholds[j]:
                    v_leg = v_leg + 1
                    filtered_images.append(i)
                else: 
                    v_adv = v_adv + 1
            
            ans = 1 if v_leg > v_adv else 0
            # helpers.assign_confusion_matrix(cm, y[i], ans)
            y_pred[i] = ans
    
    return y_pred, np.asarray(filtered_images)
