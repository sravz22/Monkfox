let todoItemsContainer = document.getElementById("todoItemsContainer");
let addTodoButton = document.getElementById("addTodoButton");
let clearItemsButton = document.getElementById("clearItemsButton");

let todoList = [{
        text: "Learn NodeJs",
        uniqueNo: 1
    },
    {
        text: "Learn ReactJs",
        uniqueNo: 2
    }, {
        text: "Learn GraphQL",
        uniqueNo: 3
    }
];

function onAddTodo() {
    let userInputEl = document.getElementById("todoUserInput");
    let userInputVal = userInputEl.value;
    let todoCount = todoList.length;
    todoCount += 1;

    if (userInputVal === "") {
        alert("enter a valid Input");
        return;
    }

    let newTodo = {
        text: userInputVal,
        uniqueNo: todoCount
    };

    todoList.push(newTodo);

    createAndAppendTodo(newTodo);
    userInputEl.value = " ";
}

addTodoButton.onclick = function() {
    onAddTodo();
};

function onDeleteTodo(todoId, todoNum) {
    let todoEl = document.getElementById(todoId);
    todoItemsContainer.removeChild(todoEl);
    todoList.splice(todoNum - 1, 1);

}

function createAndAppendTodo(todo) {
    let todoId = "todo" + todo.uniqueNo;

    let todoEl = document.createElement("li");
    todoEl.classList.add("todo-item-container", "d-flex", "flex-row");
    todoEl.id = todoId;
    todoItemsContainer.appendChild(todoEl);

    let textContainer = document.createElement("div");
    textContainer.classList.add("label-container", "d-flex", "flex-row");
    todoEl.appendChild(textContainer);

    let todoItemText = document.createElement('p');
    todoItemText.textContent = todo.text;
    textContainer.appendChild(todoItemText);

    let iconContainer = document.createElement("div");
    iconContainer.classList.add("icon-container", "d-flex", "flex-row");
    todoEl.appendChild(iconContainer);

    let checkIconEl = document.createElement("i");
    checkIconEl.classList.add("far", "fa-check-circle", "check-icon", "icon");
    iconContainer.appendChild(checkIconEl);

    let editIconEl = document.createElement("i");
    editIconEl.classList.add("far", "fa-edit", "edit-icon", "icon");
    iconContainer.appendChild(editIconEl);

    let deleteIconEl = document.createElement("i");
    deleteIconEl.classList.add("far", "fa-times-circle", "delete-icon", "icon");
    deleteIconEl.onclick = function() {
        onDeleteTodo(todoId, todo.uniqueNo);
    };
    iconContainer.appendChild(deleteIconEl);
}

for (let todo of todoList) {
    createAndAppendTodo(todo);
}

clearItemsButton.onclick = function() {
    for (let todo of todoList) {
        let todoId = 'todo' + todo.uniqueNo;
        let todoEl = document.getElementById(todoId);
        todoItemsContainer.removeChild(todoEl);
    }
    todoList = [];
};
