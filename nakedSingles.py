# Solves all the naked singles on the sudoku instance given so it can find a 
# solution faster.
def nakedSingles(instance):
  # create a 9x9x9 data structure that saves all possible values
  # for the instance
  square_helper = [[[0,1,2],[0,1,2]], [[0,1,2],[3,4,5]], [[0,1,2],[6,7,8]],
                   [[3,4,5],[0,1,2]], [[3,4,5],[3,4,5]], [[3,4,5],[6,7,8]],
                   [[6,7,8],[0,1,2]], [[6,7,8],[3,4,5]], [[6,7,8],[6,7,8]]]
  rows = 0
  columns = 1
  solution = instance
  possibilities = []
  for _ in range(1,10):
    row = []
    for _ in range(1,10):
      a = [i for i in range(1,10)]
      row.append(a)
    possibilities.append(row)

  # solve rows
  for i in range(9):
    # generates all possible values for determined row
    row_values = set(instance[i])
    if 0 in row_values:
      row_values.remove(0)
    
    # removes this values for all elements in the row
    for j in range(9):
      for elem in row_values:
        possibilities[i][j].remove(elem)

  #solve columns
  for j in range(9):
    # generates all possible values for determined column
    column_values = []
    for i in range(9):
      column_values.append(instance[i][j])
    column_values = set(column_values)
    if 0 in column_values:
      column_values.remove(0)

    # removes this values for all elements in the column
    for i in range(9):
      for elem in column_values:
        if elem in possibilities[i][j]:
          possibilities[i][j].remove(elem)

  for square in range(9):
    square_values = []
    for i in square_helper[square][rows]:
      for j in square_helper[square][columns]:
        square_values.append(instance[i][j])
    square_values = set(square_values)

    for i in square_helper[square][rows]:
      for j in square_helper[square][columns]:
        for elem in square_values:
          if elem in possibilities[i][j]:
            possibilities[i][j].remove(elem)
    
  # put naked singles in solution
  for i in range(9):
    for j in range(9):
      if len(possibilities[i][j]) == 1 and solution[i][j] == 0:
        solution[i][j] = possibilities[i][j][0]
  
  return solution
