from operator import itemgetter



def reconstruct_rating(file):
    myfile = open(file, 'r')
    user_index = {}
    restaurant_index = {}
    season_index = {'spring': 0, 'summer': 1, 'autumn': 2, 'winter': 3}
    reveiw_index = {}
    u_index = 0
    r_index = 0
    re_index = 0
    rating_list = []
    for line in myfile.readlines():
        line_array = line.strip().split(',')
        user = str(line_array[0])
        restaurant = str(line_array[1])
        rating = int(line_array[2])
        season = int(line_array[3])
        review = str(line_array[4])
        if user not in user_index.keys():
            user_index[user] = u_index
            u_index += 1

        if restaurant not in restaurant_index.keys():
            restaurant_index[restaurant] = r_index
            r_index += 1

        if review not in reveiw_index.keys():
            reveiw_index[review] = re_index
            re_index += 1
        ## the format of rating list:  (u_index, r_index, season, re_index, rating)
        rating_list.append((user_index[user], restaurant_index[restaurant], season, reveiw_index[review], rating))

    return rating_list, user_index, restaurant_index, season_index, reveiw_index


if __name__ == '__main__':
    file = "file_package/RichmondHill_filtered.txt"
    rating_list, user_index, restaurant_index, season_index, reveiw_index = reconstruct_rating(file)
    print(sorted(rating_list, key=itemgetter(3)))