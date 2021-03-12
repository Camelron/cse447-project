

with open("./reviews.csv", "r", encoding='utf-8') as f:
    with open("./output_corpus.txt", "w+", encoding='utf-8') as o:
        head = [next(f) for x in range(1, 280000, 5)]
        head = [h.split(",")[2] + '\n' for h in head]
        o.writelines(head)
        # tokens = head.split(",")
        # if (len(tokens[2]) > 1):
        #     o.write(tokens[2] + '\n')