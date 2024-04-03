import matplotlib.pyplot as plt


results_by_count = {
    "1" : 0.7174825174825175,
    "2" : 0.4381188118811881,
    "3" : 0.3237463126843656,
    "4" : 0.27141057934508817,
    "6-10" : 0.14541913341913357,
    "11-20" : 0.07825571585522959,
    "20+" : 0.028874866355962633
}
num_objects = list(results_by_count.keys())
precision_values = list(results_by_count.values())

fig = plt.figure(figsize = (10, 5))
plt.bar(num_objects, precision_values, color ='maroon', width = 0.4)
 
plt.xlabel("Number of Objects")
plt.ylabel("mAP")
plt.title("Mean Average Precision (mAP) by the number of objects in the image")
plt.savefig("../map_num_objects.png")
#############################


results_by_class = {"rare" : 0.26334296724470135, "common" : 0.21969883235886004}
classes = ["rare", "common"]
precision_values = list(results_by_class.values())
fig = plt.figure(figsize = (10, 5))
plt.bar(classes, precision_values, color = 'orange', width = 0.4)
 
plt.xlabel("Category class")
plt.ylabel("mAP")
plt.title("Mean Average Precision (mAP) by the number of objects in the image")
plt.savefig("../map_category_class.png")
